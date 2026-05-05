"""Black-box optimization problems for Part 4: Bayesian Optimization in Action.

This file contains a number of blackbox problems to optimize.

Each problem is a class exposes a consistent interface:

    .dim                   int    – number of hyperparameters (input dimension)
    .parameter_names       list   – human-readable name for each dimension
    .discrete_indices      dict   – {dim_index: 1-D array of feasible [0,1] values}
                                    empty dict means fully continuous
    .get_fixed_features_list()    – cartesian product of all discrete options,
                                    ready for botorch's optimize_acqf_mixed
    .__call__(x)           callable – x is a 1-D tensor in [0, 1]^dim,
                                      returns a scalar tensor (loss to minimise)

Available problems
------------------
Branin
    2-D, fully continuous, instant (~μs/eval).
    Classic synthetic benchmark with three global minima (f* ≈ 0.3979).
    Good for verifying that a BO loop works correctly before moving to
    expensive real-world problems.

SVMWineQuality
    2-D, fully continuous, fast (~ms/eval).
    Tunes C and γ of an SVR in log-scale over the white-wine quality dataset.

SVMClassifierWineType
    2-D, fully continuous, fast (~ms/eval).
    Tunes C and γ of an SVC that classifies red vs. white wine.
    Note: the two classes are well-separated (~99% peak accuracy), so the
    error landscape is relatively flat.

    This has a `hard` argument that removes some of the most important features
    to make the problem more interesting.

ANNWineQuality
    5-D, mixed discrete/continuous, slow (~seconds/eval).
    Tunes depth, width, dropout, learning-rate (log-scale), and training iterations
    of a small MLP over the white-wine dataset.

GradientBoostingWineQuality
    4-D, mixed discrete/continuous, medium (~100 ms/eval).
    Tunes n_estimators, max_depth, learning-rate (log-scale), and subsample
    of sklearn's GradientBoostingRegressor over the white-wine dataset.
    Interactions between depth and learning-rate create a structured landscape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import itertools
import os
import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC

_DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH  = os.path.join(_DATA_DIR, "winequality-white.csv")
_DATA_PATH_RED = os.path.join(_DATA_DIR, "winequality-red.csv")


# --------------------------------------------------------------------------- #
#  Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _load_wine(n_samples: int = 500, seed: int = 42):
    """Load, split, and return the white-wine dataset as numpy arrays."""
    df = pd.read_csv(_DATA_PATH)  # comma-delimited
    X = df.values[:n_samples, :-1]
    y = df.values[:n_samples, -1]
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)


class BlackBoxBase(ABC):
    """Mixin that auto-generates fixed_features_list from discrete_indices."""


    dim: int
    parameter_names: list[str]
    discrete_indices: dict  # subclasses must define this
    verbose: bool = False

    def get_fixed_features_list(self) -> list[dict]:
        """
        Returns the cartesian product of all discrete options as a list of dicts,
        ready to pass to botorch's optimize_acqf_mixed as fixed_features_list.
        """
        if not self.discrete_indices:
            return []
        indices = sorted(self.discrete_indices.keys())
        options = [self.discrete_indices[i] for i in indices]
        return [dict(zip(indices, combo)) for combo in itertools.product(*options)]

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

# --------------------------------------------------------------------------- #
#  Synthetic benchmark: Branin — 2-D, fully continuous                        #
# --------------------------------------------------------------------------- #

class Branin(BlackBoxBase):
    """
    The Branin-Hoo function — a classic 2-D synthetic benchmark.

    Standard form:
        f(x1, x2) = a*(x2 - b*x1^2 + c*x1 - r)^2 + s*(1-t)*cos(x1) + s

    with a=1, b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π).

    Domain and mapping from [0, 1]²:
        x1 : -5  + x[0] * 15   →   [-5, 10]
        x2 :  0  + x[1] * 15   →   [ 0, 15]

    Three global minima at f* ≈ 0.3979:
        (x1, x2) ≈ (-π, 12.275), (π, 2.275), (9.4248, 2.475)

    No __init__ arguments — this is a pure mathematical function with no
    dataset to load.
    """

    dim = 2
    parameter_names = ["x1", "x2"]
    discrete_indices: dict = {}   # fully continuous

    _a = 1.0
    _b = 5.1 / (4 * np.pi ** 2)
    _c = 5.0 / np.pi
    _r = 6.0
    _s = 10.0
    _t = 1.0 / (8 * np.pi)

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose

    def __call__(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.detach().cpu().squeeze()
        x1 = -5.0 + x[0].item() * 15.0
        x2 =  0.0 + x[1].item() * 15.0

        val = (self._a * (x2 - self._b * x1**2 + self._c * x1 - self._r)**2
               + self._s * (1 - self._t) * np.cos(x1)
               + self._s)
        if self.verbose:
            print(f"x1: {x1:.4f}  x2: {x2:.4f}  f: {val:.4f}")
        return torch.tensor(val)


# --------------------------------------------------------------------------- #
#  Problem 1: SVM (SVR) — 2-D, fully continuous                               #
# --------------------------------------------------------------------------- #

class SVMWineQuality(BlackBoxBase):
    """
    Optimise C and γ of a Support Vector Regressor (RBF kernel) on the
    white-wine quality dataset.

    Both hyperparameters span several orders of magnitude and are mapped
    from [0, 1] via a log-uniform transform:

        C     : 10^(-2 + x[0] * 4)   →   [0.01, 100]
        gamma : 10^(-4 + x[1] * 4)   →   [0.0001, 1]

    The resulting 2-D landscape is highly structured with a narrow ridge of
    good performance — the textbook example of where BO beats random search.
    The low evaluation cost also makes it easy to visualise the landscape and
    run many comparisons.
    """

    dim = 2
    parameter_names = ["log C", "log γ"]
    discrete_indices: dict = {}   # fully continuous

    _C_log_bounds     = (-2.0, 2.0)   # log10: [0.01, 100]
    _gamma_log_bounds = (-4.0, 0.0)   # log10: [0.0001, 1]

    def __init__(
            self,
            n_samples: int = 500,
            *,
            verbose: bool = False,
    ):
        self._SVR = SVR
        X_train, X_test, y_train, y_test = _load_wine(n_samples)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test  = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test  = y_test
        self.verbose = verbose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().squeeze()
        lo_C,  hi_C  = self._C_log_bounds
        lo_g,  hi_g  = self._gamma_log_bounds
        C     = 10 ** (lo_C + x[0].item() * (hi_C - lo_C))
        gamma = 10 ** (lo_g + x[1].item() * (hi_g - lo_g))

        if self.verbose:
            print(f"C: {C:.3e}  gamma: {gamma:.3e}", end="\t")
        model = self._SVR(kernel="rbf", C=C, gamma=gamma)
        model.fit(self.X_train, self.y_train)
        mse = float(np.mean((model.predict(self.X_test) - self.y_test) ** 2))
        if self.verbose:
            print(f"MSE: {mse:.4f}")
        return torch.tensor(mse).to(dtype=torch.float64)



# --------------------------------------------------------------------------- #
#  Problem 1b: SVM Classifier — red vs. white wine, 2-D continuous            #
# --------------------------------------------------------------------------- #

class SVMClassifierWineType(BlackBoxBase):
    """
    Optimise C and γ of an SVM (RBF kernel) that classifies whether a wine is
    red or white from its physicochemical features.

    Both hyperparameters are in log-scale:

        C     : 10^(-2 + x[0] * 4)   →   [0.01, 100]
        gamma : 10^(-4 + x[1] * 4)   →   [0.0001, 1]

    The black-box returns 1 - accuracy (so we minimise, consistent with the
    other problems).

    hard=False (default)
        All 11 features are used. The two classes are well-separated (~99% peak
        accuracy), so the landscape is relatively flat.

    hard=True
        The five most discriminative features are dropped: chlorides (idx 4),
        total sulfur dioxide (idx 6), volatile acidity (idx 1), fixed acidity
        (idx 0), and free sulfur dioxide (idx 5). Peak accuracy drops to ~99%
        but poor hyperparameters can fall to ~76%, creating a more
        interesting landscape.
    """

    dim = 2
    parameter_names = ["log C", "log γ"]
    discrete_indices: dict = {}   # fully continuous

    _C_log_bounds     = (-2.0, 2.0)   # log10: [0.01, 100]
    _gamma_log_bounds = (-4.0, 0.0)   # log10: [0.0001, 1]
    # feature indices to drop in hard mode (top-5 by RF importance)
    _hard_drop = [4, 6, 1, 0, 5]   # chlorides, total SO2, volatile acidity, fixed acidity, free SO2

    def __init__(
            self,
            n_samples_per_class: int = 500,
            hard: bool = False,
            *,
            verbose: bool = False,
    ):
        self._SVC = SVC

        white = pd.read_csv(_DATA_PATH).values[:n_samples_per_class, :-1]
        red   = pd.read_csv(_DATA_PATH_RED).values[:n_samples_per_class, :-1]

        if hard:
            keep = [i for i in range(white.shape[1]) if i not in self._hard_drop]
            white = white[:, keep]
            red   = red[:, keep]

        x_all = np.vstack([white, red])
        y_all = np.array([0] * len(white) + [1] * len(red))

        x_tr, x_te, y_tr, y_te = model_selection.train_test_split(
            x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(x_tr)
        self.X_test  = scaler.transform(x_te)
        self.y_train = y_tr
        self.y_test  = y_te
        self.verbose = verbose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().squeeze()
        lo_C, hi_C = self._C_log_bounds
        lo_g, hi_g = self._gamma_log_bounds
        C     = 10 ** (lo_C + x[0].item() * (hi_C - lo_C))
        gamma = 10 ** (lo_g + x[1].item() * (hi_g - lo_g))
        if self.verbose:
            print(f"C: {C:.3e}  gamma: {gamma:.3e}", end="\t")
        model = self._SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(self.X_train, self.y_train)
        accuracy = float(np.mean(model.predict(self.X_test) == self.y_test))
        error = 1.0 - accuracy
        if self.verbose:
            print(f"accuracy: {accuracy:.4f}  error: {error:.4f}")
        return torch.tensor(error).to(dtype=torch.float64)


# --------------------------------------------------------------------------- #
#  Problem 2: ANN (MLP) — 5-D, mixed discrete/continuous                      #
# --------------------------------------------------------------------------- #

class _ANN(torch.nn.Module):
    def __init__(self, n_hidden_layers: int, hidden_width: int, dropout_p: float):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(11, hidden_width),
            torch.nn.Dropout(dropout_p),
            torch.nn.ReLU(),
            *(
                (
                    torch.nn.Linear(hidden_width, hidden_width),
                    torch.nn.Dropout(dropout_p),
                    torch.nn.ReLU(),
                ) * n_hidden_layers
            ),
            torch.nn.Linear(hidden_width, 1),
        )

    def forward(self, x):
        return self.model(x)


class ANNWineQuality(BlackBoxBase):
    """
    Optimise five hyperparameters of a small MLP on the white-wine dataset.

    Hyperparameter mapping from [0, 1]:

        x[0]  n_hidden_layers  discrete  {0, 1, 2, 3}
        x[1]  n_neurons        discrete  {5, ..., 200}
        x[2]  dropout_p        continuous [0.0, 0.5]
        x[3]  learning_rate    continuous log-uniform [1e-4, 1e-1]
        x[4]  n_gd_iters       discrete  {100, ..., 1000}

    Learning rate is in log-scale (unlike the original linear mapping), which
    creates a more structured landscape where the optimal region is narrower
    and harder to stumble upon by random search.
    """

    dim = 5
    parameter_names = ["n_hidden_layers", "n_neurons", "dropout_p",
                       "learning_rate", "n_gd_iters"]

    discrete_indices = {
        0: np.arange(4) / 3,                       # {0, 1/3, 2/3, 1}
        1: (np.arange(5, 201) - 5) / 195,          # {0/195, ..., 195/195}
        4: (np.arange(100, 1001) - 100) / 900,     # {0/900, ..., 900/900}
    }

    def __init__(
            self,
            n_samples: int = 100,
            *,
            device: str | None = None,
            verbose: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        X_train, X_test, y_train, y_test = _load_wine(n_samples)
        self.DX_train = torch.tensor(X_train, dtype=torch.double).to(device)
        self.DX_test  = torch.tensor(X_test,  dtype=torch.double).to(device)
        self.Dy_train = torch.tensor(y_train, dtype=torch.double).to(device)
        self.Dy_test  = torch.tensor(y_test,  dtype=torch.double).to(device)
        self.verbose = verbose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().squeeze()
        n_hidden   = int(torch.floor(x[0] * 3))
        n_neurons  = int(torch.floor(x[1] * 195)) + 5
        dropout_p  = float(0.5 * x[2])
        lr         = 10 ** (-4.0 + x[3].item() * 3.0)   # log-scale [1e-4, 1e-1]
        n_iters    = int(torch.floor(x[4] * 900)) + 100
        if self.verbose:
            print(f"depth: {n_hidden}  width: {n_neurons}  "
                  f"dropout: {dropout_p:.2f}  lr: {lr:.2e}  iters: {n_iters}", end="\t")

        model     = _ANN(n_hidden, n_neurons, dropout_p).to(dtype=torch.double, device=self.device)
        loss_fn   = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(n_iters):
            pred = model(self.DX_train)
            loss = loss_fn(pred.reshape(-1), self.Dy_train.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(
                model(self.DX_test).reshape(-1),
                self.Dy_test.reshape(-1)
            )
        if self.verbose:
            print(f"MSE: {test_loss:.4f}")
        return test_loss.detach().cpu().to(dtype=torch.float64)


# --------------------------------------------------------------------------- #
#  Problem 3: Gradient Boosting — 4-D, mixed discrete/continuous              #
# --------------------------------------------------------------------------- #

class GradientBoostingWineQuality(BlackBoxBase):
    """
    Optimise four hyperparameters of sklearn's GradientBoostingRegressor on
    the white-wine dataset.

    Hyperparameter mapping from [0, 1]:

        x[0]  n_estimators   discrete  {50, 100, 200, 300, 500}
        x[1]  max_depth      discrete  {2, 3, 4, 5, 6}
        x[2]  learning_rate  continuous log-uniform [1e-3, 0.5]
        x[3]  subsample      continuous [0.4, 1.0]

    The interaction between max_depth and learning_rate creates a structured
    landscape: deep trees need a small learning rate (shrinkage) to generalise,
    while shallow trees tolerate higher rates. This interaction is hard to find
    by random search but easy for a GP to model.
    Evaluation is ~100 ms, sitting nicely between SVM (instant) and ANN (slow).
    """

    dim = 4
    parameter_names = ["n_estimators", "max_depth", "learning_rate", "subsample"]

    _n_est_options    = np.array([50, 100, 200, 300, 500])
    _depth_options    = np.array([2, 3, 4, 5, 6])

    discrete_indices = {
        0: (_n_est_options  - _n_est_options.min())  / (_n_est_options.max()  - _n_est_options.min()),
        1: (_depth_options  - _depth_options.min())  / (_depth_options.max()  - _depth_options.min()),
    }

    _lr_log_bounds = (-3.0, np.log10(0.5))  # [1e-3, 0.5]

    def __init__(
            self,
            n_samples: int = 500,
            *,
            verbose: bool = False,
    ):
        from sklearn.ensemble import GradientBoostingRegressor
        self._GBR = GradientBoostingRegressor
        X_train, X_test, y_train, y_test = _load_wine(n_samples)
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        self.verbose = verbose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().squeeze()

        n_est  = int(self._n_est_options[
            np.argmin(np.abs(self._n_est_options - (self._n_est_options.min()
                + x[0].item() * (self._n_est_options.max() - self._n_est_options.min()))))
        ])
        depth  = int(self._depth_options[
            np.argmin(np.abs(self._depth_options - (self._depth_options.min()
                + x[1].item() * (self._depth_options.max() - self._depth_options.min()))))
        ])
        lo, hi = self._lr_log_bounds
        lr       = 10 ** (lo + x[2].item() * (hi - lo))
        subsample = 0.4 + x[3].item() * 0.6     # [0.4, 1.0]

        if self.verbose:
            print(f"n_est: {n_est}  depth: {depth}  lr: {lr:.3e}  subsample: {subsample:.2f}", end="\t")

        model = self._GBR(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            random_state=0,
        )
        model.fit(self.X_train, self.y_train)
        mse = float(np.mean((model.predict(self.X_test) - self.y_test) ** 2))
        if self.verbose:
            print(f"MSE: {mse:.4f}")
        return torch.tensor(mse).to(dtype=torch.float64)