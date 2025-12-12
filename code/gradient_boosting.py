import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from dataset.dataset import get_dataloaders
from metrics_plots import (
    plot_training_curves,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_permutation_importance,
    ensure_dir,
)

class GradientBoostingEcosystem(nn.Module):
    """
    Wrapper around sklearn HistGradientBoostingClassifier (multiclass capable)
    so it can be used with the same dataloaders and evaluation utilities as FusionNet.
    Uses ONLY tabular variables (no images).
    """

    def __init__(
        self,
        num_classes: int = 17,
        var_input_dim: Optional[int] = None,
        random_state: int = 42,
        learning_rate: float = 0.05,
        max_iter: int = 400,
        max_depth: Optional[int] = 6,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.var_input_dim = var_input_dim
        self.random_state = random_state

        self.gb = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        )

        self._is_trained = False

    @staticmethod
    def _collect_xy_from_loader(loader) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for batch in loader:
            variables = batch.get("variables")
            labels = batch.get("labels")
            if variables is None or labels is None:
                continue

            mask = labels >= 0
            if mask.sum() == 0:
                continue

            xs.append(variables[mask].numpy())
            ys.append(labels[mask].numpy())

        if len(xs) == 0:
            return np.empty((0,)), np.empty((0,))

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return X, y
    
    @staticmethod
    def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
        """
        Compute per-sample weights from class frequencies.
        Uses balanced weighting: N / (K * n_c)
        """
        classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y,
        )
        weight_map = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_map[label] for label in y])
        return sample_weights

    def fit_from_loader(self, train_loader) -> None:
        X_train, y_train = self._collect_xy_from_loader(train_loader)
        if X_train.size == 0:
            raise ValueError("No variables found in train_loader to fit GradientBoosting.")

        if self.var_input_dim is not None and X_train.shape[1] != self.var_input_dim:
            print(
                f"[Warning] GB: inferred var_input_dim={X_train.shape[1]}, "
                f"but var_input_dim={self.var_input_dim} was passed."
            )

        print(f"Fitting GradientBoosting on X_train={X_train.shape}, y_train={y_train.shape}")
        # Class weights computation
        sample_weights = self._compute_sample_weights(y_train)
        print(
            "Class distribution:",
            dict(zip(*np.unique(y_train, return_counts=True)))
        )
        print(
            "Sample weight range:",
            f"{sample_weights.min():.3f} → {sample_weights.max():.3f}"
        )

        self.gb.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

        self._is_trained = True

    def fit_optimized(self, train_loader) -> None:
        """
        RandomizedSearchCV over HistGradientBoosting hyperparameters.
        """
        X_train, y_train = self._collect_xy_from_loader(train_loader)
        if X_train.size == 0:
            raise ValueError("No variables found in train_loader to fit GradientBoosting.")

        print(
            f"Finding optimized parameters for GradientBoosting on "
            f"X_train={X_train.shape}, y_train={y_train.shape}"
        )

        param_distributions = {
            "learning_rate": [0.005, 0.01, 0.03],
            "max_iter": [600, 800],
            "max_depth": [None],
            "max_leaf_nodes": [63, 101, 203],
            "min_samples_leaf": [50, 60],
            "l2_regularization": [0.0, 1e-2, 0.1, 0.2],
        }

        search = RandomizedSearchCV(
            estimator=self.gb,
            param_distributions=param_distributions,
            n_iter=20,
            cv=3,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )

        sample_weights = self._compute_sample_weights(y_train)

        search.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

        print(f"Best parameters: {search.best_params_}")

        self.gb = search.best_estimator_
        self._is_trained = True

    def permutation_importance(self, loader, n_repeats: int = 30) -> pd.DataFrame:
        X, y = self._collect_xy_from_loader(loader)
        if X.size == 0:
            raise ValueError("No variables found in loader for permutation importance.")

        var_names = loader.dataset.var_cols
        print(f"Computing permutation importance on X={X.shape}, y={y.shape}")

        perm_imp = permutation_importance(
            self.gb,
            X,
            y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="accuracy",
        )

        sorted_idx = perm_imp.importances_mean.argsort()
        importances = pd.DataFrame(
            perm_imp.importances[sorted_idx].T,
            columns=[var_names[i] for i in sorted_idx],
        )
        return importances

    def forward(
        self,
        images: Optional[torch.Tensor],
        variables: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("GradientBoostingEcosystem must be fitted before calling forward().")
        if variables is None:
            raise ValueError("GradientBoostingEcosystem requires 'variables' as input.")

        X = variables.cpu().numpy()
        proba = self.gb.predict_proba(X)

        eps = 1e-9
        logits_np = np.log(proba + eps)
        logits = torch.from_numpy(logits_np).to(variables.device)
        return logits
    
def evaluate(model, loader, device, loss_fn):
    """
    Generic evaluation loop:
    - calls model(images, variables) → logits
    - computes loss and accuracy
    Works for both FusionNet and RandomForestEcosystem.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            images = batch.get("images")
            variables = batch.get("variables")
            labels = batch.get("labels")
            if labels is None:
                continue

            if images is not None:
                images = images.to(device)
            if variables is not None:
                variables = variables.to(device)
            labels = labels.to(device)

            logits = model(images, variables)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(batch_size)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc

    
def main():
    parser = argparse.ArgumentParser(description="RandomForest baseline on SWECO variables")
    parser.add_argument(
        "--variable-selection",
        dest="variable_selection",
        nargs="*",
        default=["all"],
        help="Variable selection: 'all', a group key, or multiple keys (e.g., geol edaph)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="gb_exp",
        help="Output directory for results",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parents[1].resolve() / "data"
    csv_path = data_dir / "dataset_split.csv"
    image_dir = data_dir / "images"
    variable_selection = args.variable_selection

    ### Hyperparams ###
    batch_size = 256         # batching only affects how we iterate to collect data
    num_workers = 6
    num_classes = 17
    label_smoothing = 0.0    # for Gradient boosting logits, no need for smoothing

    # Gradient boosting is variables-only → disable image loading
    loaders = get_dataloaders(
        csv_path=csv_path,
        image_dir=image_dir,
        variable_selection=variable_selection,
        batch_size=batch_size,
        num_workers=num_workers,
        load_images=False,
    )

    # Infer variable input dimension from one batch
    sample_batch = next(iter(loaders["train"]))
    var_tensor = sample_batch.get("variables")
    var_input_dim = var_tensor.shape[1] if var_tensor is not None else None
    print(f"Variables used for this run: {variable_selection}")
    print(f"Variable dimension: {var_input_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GradientBoostingEcosystem(
        num_classes=num_classes,
        var_input_dim=var_input_dim,
        learning_rate=0.01,
        max_iter=900,
        max_depth=None,
        min_samples_leaf=50,
        l2_regularization=0.0,
    )
    
    # ---- Train Gradient boosting on train loader ----
    model.fit_from_loader(loaders["train"])

    # ---- Find best parameters with RandomizedSearchCV and train Gradient boosting on train loader ----
    optimize_params = False
    if (optimize_params) :
        model.fit_optimized(loaders["train"])

    # Best parameters : {'min_samples_leaf': 50, 'max_leaf_nodes': 203, 'max_iter': 800, 'max_depth': None, 'learning_rate': 0.01, 'l2_regularization': 0.0}

    # ---- Evaluate on train / val / test ----
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_loss, train_acc = evaluate(model, loaders["train"], device, loss_fn)
    val_loss, val_acc = evaluate(model, loaders["val"], device, loss_fn)
    test_loss, test_acc = evaluate(model, loaders["test"], device, loss_fn)

    print(f"Train | loss {train_loss:.4f} acc {train_acc:.3f}")
    print(f"Val   | loss {val_loss:.4f} acc {val_acc:.3f}")
    print(f"Test  | loss {test_loss:.4f} acc {test_acc:.3f}")

    # Save results to csv
    results_dir = f"results/{args.out_dir}"
    ensure_dir(results_dir)
    results_df = pd.DataFrame([[train_loss, train_acc], [val_loss, val_acc], [test_loss, test_acc]], 
                              index=["Train", "Val", "Test"], columns=["loss", "acc"])
    results_df_path = os.path.join(results_dir, "results.csv")
    results_df.to_csv(results_df_path)
    print(f"Saved loss/acc results to {results_df_path}")


    # For compatibility with plot_training_curves, create a "1-epoch" metrics dict
    metrics = {
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "train_acc": [train_acc],
        "val_acc": [val_acc],
    }

    # Plotting and confusion matrix generation

    curves_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(metrics, curves_path)
    print(f"Saved training curves to {curves_path}")

    # You can choose val or test for confusion matrix; here I use test
    cm_tensor, class_names = compute_confusion_matrix(model, loaders["test"], device)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm_tensor, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # ---- Permutation importance computation and plotting ----
    importances = model.permutation_importance(loaders["test"], n_repeats=30)

    plot_permutation_importance(importances, results_dir)

if __name__ == "__main__":
    main()
