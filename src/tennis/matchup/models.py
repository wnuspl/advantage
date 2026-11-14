"""Matchup models. Every model outputs delta in logit space on top of a frozen offset:

    logit P(p1 wins) = offset + delta,   offset = alpha * glicko_logit

and every delta is antisymmetric BY CONSTRUCTION: delta(a, b) = -delta(b, a), so
P(a beats b) = 1 - P(b beats a). Rows where either player has no style vector get
delta = 0 exactly (Glicko-only), whatever the model.

Pairing (spec §5): p1 serve x p2 return, p1 return x p2 serve, p1 rally x p2 rally.

Interface (spec §7):  m.fit(train, val);  m.predict_proba(test) -> P(p1 wins)
where train/val/test are data.FoldData of one tour and one fold.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import FoldData, sigmoid

SIDES = ("serve", "return", "rally")


class MatchupModel:
    name = "base"

    def fit(self, train: FoldData, val: FoldData) -> None:
        raise NotImplementedError

    def delta(self, fd: FoldData) -> np.ndarray:
        """Antisymmetric logit adjustment for rows where both players have vectors."""
        raise NotImplementedError

    def predict_proba(self, test: FoldData) -> np.ndarray:
        d = np.zeros(len(test))
        if test.has.any():
            d[test.has] = self.delta(test.subset(test.has))
        return sigmoid(test.offset + d)

    def config(self) -> dict:
        return {}

    def state(self):
        """Picklable trained state (saved in the bundle with fit_ids and alpha)."""
        return None

    info: dict = {}


class GlickoOnly(MatchupModel):
    """The benchmark: delta = 0, P = sigmoid(alpha * glicko_logit)."""
    name = "glicko"

    def fit(self, train, val):
        self.info = {}

    def delta(self, fd):
        return np.zeros(len(fd))


# ------------------------------------------------------------------ torch models

@dataclass(frozen=True)
class TrainConfig:
    lr: float = 0.01
    weight_decay: float = 0.0       # AdamW decoupled decay on every parameter
    l2: float = 0.0                 # explicit penalty (bilinear: on the effective W matrices)
    batch_size: int = 0             # 0 = full batch
    max_epochs: int = 400
    patience: int = 25              # epochs without val improvement before stopping
    min_epochs: int = 5
    seed: int = 0


def _torch():
    import torch
    return torch


def _tensors(fd: FoldData) -> dict:
    torch = _torch()
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)  # noqa: E731
    out = {"y": t(fd.y), "offset": t(fd.offset), "surf": t(fd.surf)}
    for s in SIDES:
        out["a_" + s], out["b_" + s] = t(fd.xa[s]), t(fd.xb[s])
    return out


def _take(batch: dict, idx) -> dict:
    return {k: v[idx] for k, v in batch.items()}


def train_torch(module, train: FoldData, val: FoldData, cfg: TrainConfig) -> dict:
    """Minimize BCE(offset + delta, y) + penalty with AdamW; early stop on val log loss and
    restore the best epoch. Returns training info."""
    torch = _torch()
    torch.manual_seed(cfg.seed)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    tr, va = _tensors(train), (_tensors(val) if len(val) else None)
    n = len(train)
    opt = torch.optim.AdamW(module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    bs = cfg.batch_size or n

    def val_loss():
        module.eval()
        with torch.no_grad():
            return float(bce(va["offset"] + module(va), va["y"]))

    best = (val_loss() if va else math.inf, copy.deepcopy(module.state_dict()), 0)
    start_val, bad, epoch = best[0], 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        module.train()
        perm = torch.randperm(n, generator=gen) if bs < n else torch.arange(n)
        for i in range(0, n, bs):
            b = _take(tr, perm[i:i + bs])
            loss = bce(b["offset"] + module(b), b["y"]) + module.penalty() * len(b["y"]) / n
            opt.zero_grad()
            loss.backward()
            opt.step()
        if va is None:
            continue
        vl = val_loss()
        if vl < best[0] - 1e-7:
            best, bad = (vl, copy.deepcopy(module.state_dict()), epoch), 0
        else:
            bad += 1
            if bad >= cfg.patience and epoch >= cfg.min_epochs:
                break
    if va is not None:
        module.load_state_dict(best[1])
    module.eval()
    return {"epochs": epoch, "best_epoch": best[2], "val_loss_start": start_val,
            "val_loss_best": best[0], "n_train": n, "n_val": len(val)}


class TorchModel(MatchupModel):
    def __init__(self, train_cfg: TrainConfig = TrainConfig()):
        self.train_cfg = train_cfg
        self.module = None
        self.dims = None

    def build(self, dims: dict):
        raise NotImplementedError

    def fit(self, train, val):
        self.dims = train.dims()
        self.module = self.build(self.dims)
        tr, va = train.subset(train.has), val.subset(val.has)
        self.info = train_torch(self.module, tr, va, self.train_cfg) if len(tr) else {"n_train": 0}

    def delta(self, fd):
        torch = _torch()
        if self.module is None:
            raise RuntimeError("fit() first")
        self.module.eval()
        with torch.no_grad():
            return self.module(_tensors(fd)).numpy().astype(float)

    def state(self):
        return {"dims": self.dims, "state_dict": None if self.module is None else self.module.state_dict()}


def _bilinear_module(dims: dict, rank: int, terms: tuple, l2: float, seed: int):
    torch = _torch()
    nn = torch.nn

    class Bilinear(nn.Module):
        """delta = a_s W_sr b_r - b_s W_sr a_r + a_l W_rr b_l,
        W_sr = U V^T (low rank), W_rr = P Q^T - Q P^T (antisymmetric, so the rally term
        flips sign when players swap). V and Q start at zero: delta = 0 at init."""

        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            ks, kr, kl = dims["serve"], dims["return"], dims["rally"]
            self.U = nn.Parameter(0.1 * torch.randn(ks, rank, generator=g))
            self.V = nn.Parameter(torch.zeros(kr, rank))
            self.P = nn.Parameter(0.1 * torch.randn(kl, rank, generator=g))
            self.Q = nn.Parameter(torch.zeros(kl, rank))

        def w_sr(self):
            return self.U @ self.V.T

        def w_rr(self):
            return self.P @ self.Q.T - self.Q @ self.P.T

        def forward(self, b):
            d = torch.zeros(len(b["y"]))
            if "sr" in terms:
                W = self.w_sr()
                d = d + ((b["a_serve"] @ W) * b["b_return"]).sum(-1) \
                      - ((b["b_serve"] @ W) * b["a_return"]).sum(-1)
            if "rr" in terms:
                d = d + ((b["a_rally"] @ self.w_rr()) * b["b_rally"]).sum(-1)
            return d

        def penalty(self):
            p = torch.zeros(())
            if "sr" in terms:
                p = p + (self.w_sr() ** 2).sum()
            if "rr" in terms:
                p = p + (self.w_rr() ** 2).sum()
            return l2 * p

    return Bilinear()


class BilinearModel(TorchModel):
    """Antisymmetric bilinear interaction (spec §5.1). terms: 'sr' serve-vs-return,
    'rr' rally-vs-rally; ablate with terms=('sr',) or ('rr',)."""

    def __init__(self, rank: int = 3, terms: tuple = ("sr", "rr"),
                 train_cfg: TrainConfig = TrainConfig(lr=0.02, l2=1e-3, max_epochs=600, patience=40)):
        super().__init__(train_cfg)
        self.rank, self.terms = rank, tuple(terms)
        self.name = "bilinear" + ("" if set(self.terms) == {"sr", "rr"} else "_" + "_".join(self.terms))

    def build(self, dims):
        return _bilinear_module(dims, self.rank, self.terms, self.train_cfg.l2, self.train_cfg.seed)

    def config(self):
        return {"rank": self.rank, "terms": list(self.terms), "train": asdict(self.train_cfg)}


def _siamese_module(dims: dict, hidden: int, head_hidden: int, dropout: float, n_surf: int, seed: int):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    k_total = dims["serve"] + dims["return"] + dims["rally"]

    class SiameseMatchup(nn.Module):
        """Shared encoder g for both players; d = h(e_a, e_b, surf) - h(e_b, e_a, surf).

        Spec §5.2 has a linear h. A linear h reduces to (w1 - w2).(e_a - e_b): a per-player
        score difference with no interaction and the surface cancelling out. head_hidden > 0
        makes h a small MLP so pairings and surface can matter (head_hidden = 0 = spec as
        written). h's last layer starts at zero: delta = 0 (Glicko-only) before training."""

        def __init__(self):
            super().__init__()
            self.g = nn.Sequential(nn.Linear(k_total, hidden), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.ReLU())
            if head_hidden:
                self.h = nn.Sequential(nn.Linear(2 * hidden + n_surf, head_hidden), nn.ReLU(),
                                       nn.Linear(head_hidden, 1))
            else:
                self.h = nn.Sequential(nn.Linear(2 * hidden + n_surf, 1))
            nn.init.zeros_(self.h[-1].weight)
            nn.init.zeros_(self.h[-1].bias)

        def forward(self, b):
            xa = torch.cat([b["a_serve"], b["a_return"], b["a_rally"]], -1)
            xb = torch.cat([b["b_serve"], b["b_return"], b["b_rally"]], -1)
            ea, eb = self.g(xa), self.g(xb)
            s = b["surf"]
            d = self.h(torch.cat([ea, eb, s], -1)) - self.h(torch.cat([eb, ea, s], -1))
            return d.squeeze(-1)

        def penalty(self):
            return torch.zeros(())

    return SiameseMatchup()


class SiameseModel(TorchModel):
    """Siamese antisymmetric MLP (spec §5.2)."""
    name = "siamese"

    def __init__(self, hidden: int = 32, head_hidden: int = 16, dropout: float = 0.2,
                 train_cfg: TrainConfig = TrainConfig(lr=3e-3, weight_decay=1e-2, batch_size=1024,
                                                      max_epochs=60, patience=6)):
        super().__init__(train_cfg)
        self.hidden, self.head_hidden, self.dropout = hidden, head_hidden, dropout

    def build(self, dims):
        return _siamese_module(dims, self.hidden, self.head_hidden, self.dropout, 3, self.train_cfg.seed)

    def config(self):
        return {"hidden": self.hidden, "head_hidden": self.head_hidden, "dropout": self.dropout,
                "train": asdict(self.train_cfg)}


# ------------------------------------------------------------------ LightGBM

@dataclass(frozen=True)
class LGBMConfig:
    num_leaves: int = 15
    learning_rate: float = 0.03
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 10.0
    num_boost_round: int = 2000
    early_stopping: int = 100
    seed: int = 0


def lgbm_features(fd: FoldData, swap: bool = False) -> np.ndarray:
    """glicko_logit, both players' vectors, paired products, surface. Serve and return
    vectors can have different lengths: products use their first min(K_serve, K_return)
    components."""
    a, b = (fd.xb, fd.xa) if swap else (fd.xa, fd.xb)
    k = min(a["serve"].shape[1], a["return"].shape[1])
    gl = -fd.glicko_logit if swap else fd.glicko_logit
    return np.hstack([gl[:, None], a["serve"], a["return"], a["rally"], b["serve"], b["return"], b["rally"],
                      a["serve"][:, :k] * b["return"][:, :k], a["return"][:, :k] * b["serve"][:, :k],
                      a["rally"] * b["rally"], fd.surf])


class LGBMModel(MatchupModel):
    """Gradient-boosted benchmark (spec §5.3). Trained on both orderings with the frozen
    offset as init_score; prediction averages p(a,b) and 1 - p(b,a), so it is symmetric."""
    name = "lgbm"

    def __init__(self, cfg: LGBMConfig = LGBMConfig()):
        self.cfg = cfg
        self.booster = None

    def _params(self):
        c = self.cfg
        return {"objective": "binary", "num_leaves": c.num_leaves, "learning_rate": c.learning_rate,
                "min_data_in_leaf": c.min_data_in_leaf, "feature_fraction": c.feature_fraction,
                "bagging_fraction": c.bagging_fraction, "bagging_freq": 1, "lambda_l2": c.lambda_l2,
                "seed": c.seed, "deterministic": True, "force_row_wise": True, "verbosity": -1,
                "num_threads": 0}

    @staticmethod
    def _both(fd: FoldData):
        X = np.vstack([lgbm_features(fd), lgbm_features(fd, swap=True)])
        y = np.concatenate([fd.y, 1.0 - fd.y])
        off = np.concatenate([fd.offset, -fd.offset])
        return X, y, off

    def fit(self, train, val):
        import lightgbm as lgb
        tr, va = train.subset(train.has), val.subset(val.has)
        if not len(tr):
            self.info = {"n_train": 0}
            return
        X, y, off = self._both(tr)
        dtr = lgb.Dataset(X, y, init_score=off, free_raw_data=False)
        kw, cb = {}, []
        if len(va):
            Xv, yv, offv = self._both(va)
            kw["valid_sets"] = [lgb.Dataset(Xv, yv, init_score=offv, reference=dtr)]
            cb.append(lgb.early_stopping(self.cfg.early_stopping, verbose=False))
        self.booster = lgb.train(self._params(), dtr, num_boost_round=self.cfg.num_boost_round,
                                 callbacks=cb, **kw)
        self.info = {"best_iteration": self.booster.best_iteration, "n_train": len(tr), "n_val": len(va)}

    def _raw(self, X):
        it = self.booster.best_iteration or None
        return self.booster.predict(X, raw_score=True, num_iteration=it)

    def predict_proba(self, test):
        p = sigmoid(test.offset)
        if self.booster is None or not test.has.any():
            return p
        fd = test.subset(test.has)
        p_ab = sigmoid(fd.offset + self._raw(lgbm_features(fd)))
        p_ba = sigmoid(-fd.offset + self._raw(lgbm_features(fd, swap=True)))
        p[test.has] = 0.5 * (p_ab + 1.0 - p_ba)
        return p

    def delta(self, fd):
        """Logit of the symmetrized probability minus the offset (antisymmetric)."""
        p = self.predict_proba(fd)
        return np.log(p) - np.log1p(-p) - fd.offset

    def config(self):
        return asdict(self.cfg)

    def state(self):
        return None if self.booster is None else self.booster.model_to_string()


# ------------------------------------------------------------------ registry

def make(name: str) -> MatchupModel:
    """Fresh model by name: glicko, bilinear, bilinear_sr, bilinear_rr, siamese, lgbm."""
    factories = {
        "glicko": GlickoOnly,
        "bilinear": lambda: BilinearModel(terms=("sr", "rr")),
        "bilinear_sr": lambda: BilinearModel(terms=("sr",)),
        "bilinear_rr": lambda: BilinearModel(terms=("rr",)),
        "siamese": SiameseModel,
        "lgbm": LGBMModel,
    }
    if name not in factories:
        raise ValueError(f"unknown matchup model {name!r}; known: {sorted(factories)}")
    return factories[name]()


MODEL_NAMES = ("glicko", "bilinear", "bilinear_sr", "bilinear_rr", "siamese", "lgbm")
