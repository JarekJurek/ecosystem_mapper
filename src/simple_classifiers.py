import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from dataset.dataset import get_dataloaders
from metrics_plots import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_permutation_importance,
)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class SklearnEcosystemModel(nn.Module):
    def __init__(self, estimator, num_classes, var_input_dim=None):
        super().__init__()
        self.estimator = estimator
        self.num_classes = num_classes
        self.var_input_dim = var_input_dim
        self._is_trained = False

    @staticmethod
    def collect_xy(loader):
        xs, ys = [], []
        for batch in loader:
            v = batch.get("variables")
            y = batch.get("labels")
            if v is None or y is None:
                continue
            mask = y >= 0
            if mask.sum() == 0:
                continue
            xs.append(v[mask].numpy())
            ys.append(y[mask].numpy())
        return np.concatenate(xs), np.concatenate(ys)

    def fit_from_loader(self, train_loader):
        X, y = self.collect_xy(train_loader)
        self.estimator.fit(X, y)
        self._is_trained = True

    def forward(self, images, variables):
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        X = variables.cpu().numpy()
        proba = self.estimator.predict_proba(X)
        logits = np.log(proba + 1e-9)
        return torch.from_numpy(logits).to(variables.device)

    def permutation_importance(self, loader, n_repeats=30):
        X, y = self.collect_xy(loader)
        var_names = loader.dataset.var_cols
        perm = permutation_importance(
            self.estimator, X, y,
            n_repeats=n_repeats,
            random_state=42,
            scoring="f1_macro",
        )
        idx = perm.importances_mean.argsort()
        return pd.DataFrame(
            perm.importances[idx].T,
            columns=[var_names[i] for i in idx],
        )

def build_random_forest(var_input_dim, num_classes):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="log2",
        bootstrap=False,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return SklearnEcosystemModel(rf, num_classes, var_input_dim)

def build_gradient_boosting(var_input_dim, num_classes):
    gb = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=42,
    )
    return SklearnEcosystemModel(gb, num_classes, var_input_dim)

def optimize_random_forest(estimator, X, y):
    param_dist = {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    print(f"[RF] Best params: {search.best_params_}")
    return search.best_estimator_

def optimize_gradient_boosting(estimator, X, y):
    param_dist = {
        "learning_rate": [0.01, 0.03, 0.05],
        "max_iter": [400, 600, 800],
        "max_leaf_nodes": [31, 63, 127],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.01, 0.1],
    }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    print(f"[HGB] Best params: {search.best_params_}")
    return search.best_estimator_

def evaluate(model, loader, device, loss_fn):
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            variables = batch.get("variables")
            labels = batch.get("labels")
            if variables is None or labels is None:
                continue

            variables = variables.to(device)
            labels = labels.to(device)

            logits = model(None, variables)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == labels).sum().item()
            total_samples += bs

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    f1 = f1_score(
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
        average="macro",
    )
    return avg_loss, acc, f1

def main():
    parser = argparse.ArgumentParser(description="Tree-based baselines on SWECO variables")
    parser.add_argument("--model", choices=["rf", "gb"], default="rf")
    parser.add_argument("--variable-selection", nargs="*", default=["all"])
    parser.add_argument("--out-dir", default="sklearn_exp")
    parser.add_argument("--optimize", action="store_true",
                        help="Run RandomizedSearchCV before training")
    parser.add_argument("--perm-importance", action="store_true",
                        help="Compute and plot permutation feature importance")
    args = parser.parse_args()

    data_dir = Path(__file__).parents[1] / "data"

    loaders = get_dataloaders(
        csv_path=data_dir / "dataset_split.csv",
        image_dir=data_dir / "images",
        variable_selection=args.variable_selection,
        batch_size=256,
        num_workers=6,
        load_images=False,
    )

    sample = next(iter(loaders["train"]))
    var_input_dim = sample["variables"].shape[1]
    num_classes = 17
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "rf":
        model = build_random_forest(var_input_dim, num_classes)
    else:
        model = build_gradient_boosting(var_input_dim, num_classes)

    if args.optimize:
        X_train, y_train = model.collect_xy(loaders["train"])

        if args.model == "rf":
            model.estimator = optimize_random_forest(model.estimator, X_train, y_train)
        else:
            model.estimator = optimize_gradient_boosting(model.estimator, X_train, y_train)

    model.fit_from_loader(loaders["train"])

    loss_fn = nn.CrossEntropyLoss()

    train = evaluate(model, loaders["train"], device, loss_fn)
    val = evaluate(model, loaders["val"], device, loss_fn)
    test = evaluate(model, loaders["test"], device, loss_fn)

    print(f"Train | loss {train[0]:.3f} acc {train[1]:.3f} f1 {train[2]:.3f}")
    print(f"Val   | loss {val[0]:.3f} acc {val[1]:.3f} f1 {val[2]:.3f}")
    print(f"Test  | loss {test[0]:.3f} acc {test[1]:.3f} f1 {test[2]:.3f}")

    results_dir = f"results/{args.out_dir}_{args.model}"
    ensure_dir(results_dir)

    cm, class_names = compute_confusion_matrix(model, loaders["test"], device)
    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, "confusion_matrix.png"))

    if args.perm_importance:
        print("Computing permutation feature importance...")
        importances = model.permutation_importance(loaders["test"], n_repeats=30)
        plot_permutation_importance(importances, results_dir)

if __name__ == "__main__":
    main()
