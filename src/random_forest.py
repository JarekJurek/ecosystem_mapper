import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

class RandomForestEcosystem(nn.Module):
    """
    Wrapper around sklearn RandomForestClassifier so it can be used
    with the same dataloaders and evaluation utilities as FusionNet.

    Parameters
    ----------
    num_classes : int
        Number of ecosystem classes (default: 17).
    var_input_dim : int or None
        Dimensionality of the variable vector. Used only for sanity checks.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs for training/prediction.
    class_weight : str or dict or None
        Class weights (e.g. 'balanced_subsample') to handle imbalance.
    """

    def __init__(
        self,
        num_classes: int = 17,
        var_input_dim: Optional[int] = None,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: str = "balanced_subsample",
        min_samples_split: Optional[int] = 5,
        min_samples_leaf: Optional[int] = 2,
        max_features: Optional[int] = 'log2',
        bootstrap: Optional[bool] = False
    ) -> None:

        super().__init__()
        self.num_classes = num_classes
        self.var_input_dim = var_input_dim
        self.random_state = random_state

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap
        )

        self._is_trained = False

    @staticmethod
    def _collect_xy_from_loader(loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collects variables (X) and labels (y) from a PyTorch DataLoader
        into NumPy arrays. Assumes the collate_fn outputs keys:
        'variables' (B, D) and 'labels' (B,).
        """
        xs = []
        ys = []
        for batch in loader:
            variables = batch.get("variables")  # (B, D) or None
            labels = batch.get("labels")        # (B,) or None
            if variables is None or labels is None:
                continue

            # Filter out dummy labels (-1) just in case
            mask = labels >= 0
            if mask.sum() == 0:
                continue

            vars_np = variables[mask].numpy()
            labels_np = labels[mask].numpy()

            xs.append(vars_np)
            ys.append(labels_np)

        if len(xs) == 0:
            return np.empty((0,)), np.empty((0,))

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return X, y

    def fit_from_loader(self, train_loader, val_loader=None) -> None:
        """
        Fits the RandomForest on all data from train_loader.
        (val_loader can be passed if you want to inspect performance here,
        but it's not used internally.)

        After calling this, self._is_trained becomes True and forward() can be used.
        """
        X_train, y_train = self._collect_xy_from_loader(train_loader)
        if X_train.size == 0:
            raise ValueError("No variables found in train_loader to fit RandomForest.")

        if self.var_input_dim is not None and X_train.shape[1] != self.var_input_dim:
            print(
                f"[Warning] RF: inferred var_input_dim={X_train.shape[1]}, "
                f"but var_input_dim={self.var_input_dim} was passed."
            )

        print(f"Fitting RandomForest on X_train={X_train.shape}, y_train={y_train.shape}")
        self.rf.fit(X_train, y_train)
        self._is_trained = True

    def fit_optimized(self, train_loader) -> None:
        """
        Performs hyperparameter search for the RandomForest using RandomizedSearchCV.
        """
        X_train, y_train = self._collect_xy_from_loader(train_loader)
        if X_train.size == 0:
            raise ValueError("No variables found in train_loader to fit RandomForest.")

        if self.var_input_dim is not None and X_train.shape[1] != self.var_input_dim:
            print(
                f"[Warning] RF: inferred var_input_dim={X_train.shape[1]}, "
                f"but var_input_dim={self.var_input_dim} was passed."
            )

        print(
            f"Finding optimized parameters for RandomForest on "
            f"X_train={X_train.shape}, y_train={y_train.shape}"
        )

        param_distributions = {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

        search = RandomizedSearchCV(
            estimator=self.rf,
            param_distributions=param_distributions,
            n_iter=20,               # number of models to evaluate
            cv=3,                    # 3-fold cross-validation
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        print(f"Best parameters: {search.best_params_}")

        # Apply the best parameters to the RF
        self.rf = search.best_estimator_
        self._is_trained = True

    def permutation_importance(self, loader, n_repeats =30) -> None:
        """
        Computes features permutation importance for the RandomForest on all data from train_loader.

        """
        X, y = self._collect_xy_from_loader(loader)
        if X.size == 0:
            raise ValueError("No variables found in train_loader to fit RandomForest.")

        if self.var_input_dim is not None and X.shape[1] != self.var_input_dim:
            print(
                f"[Warning] RF: inferred var_input_dim={X.shape[1]}, "
                f"but var_input_dim={self.var_input_dim} was passed."
            )

        print(f"Computing the features permutation importance for the RandomForest on X_test={X.shape}, y_test={y.shape}")
        var_names = loader.dataset.var_cols  # list of column names (length = X.shape[1])
        perm_imp = permutation_importance(self.rf, X, y,
                                        n_repeats=n_repeats,
                                        random_state=self.random_state)

        sorted_idx = perm_imp.importances_mean.argsort()

        importances = pd.DataFrame(
            perm_imp.importances[sorted_idx].T,
            columns=[var_names[i] for i in sorted_idx]
        )

        return importances

    def forward(
        self,
        images: Optional[torch.Tensor],
        variables: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward interface for compatibility with evaluate() and compute_confusion_matrix.
        - images is ignored (RF is variables-only).
        - variables is a torch.Tensor of shape (B, D).
        Returns:
            logits: torch.Tensor of shape (B, num_classes)
        """
        if not self._is_trained:
            raise RuntimeError("RandomForestEcosystem must be fitted before calling forward().")

        if variables is None:
            raise ValueError("RandomForestEcosystem requires 'variables' as input.")

        # Move variables to CPU and convert to NumPy
        X = variables.cpu().numpy()  # (B, D)

        # Predict class probabilities (B, num_classes)
        proba = self.rf.predict_proba(X)  # sklearn always returns probs for each class
        # Convert probabilities to logits by taking log; these are valid logits since
        # softmax(log(p)) = p (up to numerical eps).
        eps = 1e-9
        logits_np = np.log(proba + eps)  # (B, num_classes)

        # Convert back to torch.Tensor on the same device as variables
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
        default="rf_exp",
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
    label_smoothing = 0.0    # for RF logits, no need for smoothing

    # RF is variables-only → disable image loading
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

    # Instantiate RF model wrapper
    model = RandomForestEcosystem(
        num_classes=num_classes,
        var_input_dim=var_input_dim,   # number of SWECO vars used
        n_estimators = 300,
        min_samples_split = 5,
        min_samples_leaf = 2,
        max_features = 'log2',
        max_depth = 20,
        bootstrap = False
    )

    # ---- Train RF on train loader ----
    model.fit_from_loader(loaders["train"])

    # ---- Find best parameters with RandomizedSearchCV and train RF on train loader ----
    optimize_params = False
    if (optimize_params) :
        model.fit_optimized(loaders["train"])

    # Best parameters : {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 20, 'bootstrap': False}

    # ---- Evaluate on train / val / test ----
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_loss, train_acc = evaluate(model, loaders["train"], device, loss_fn)
    val_loss, val_acc = evaluate(model, loaders["val"], device, loss_fn)
    test_loss, test_acc = evaluate(model, loaders["test"], device, loss_fn)

    print(f"Train | loss {train_loss:.4f} acc {train_acc:.3f}")
    print(f"Val   | loss {val_loss:.4f} acc {val_acc:.3f}")
    print(f"Test  | loss {test_loss:.4f} acc {test_acc:.3f}")

    # For compatibility with plot_training_curves, create a "1-epoch" metrics dict
    metrics = {
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "train_acc": [train_acc],
        "val_acc": [val_acc],
    }

    # Plotting and confusion matrix generation
    results_dir = f"results/{args.out_dir}"
    ensure_dir(results_dir)

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
