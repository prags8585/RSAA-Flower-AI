# fed/train_reranker_flower.py  — Federated LR reranker with DP + SecAgg-style masking
from typing import Optional
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import inspect

import re
import yaml  # pip install pyyaml

FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

REGIONS = ["EU", "US", "APAC"]
N_FEATURES = 3  # [fused_score, inv_rank, snippet_len]


# ----------------- Data -----------------
def load_xy(region: str):
    """Load per-region feedback; auto-balance if only one class is present."""
    p = Path("feedback") / f"{region}_feedback.csv"
    if not p.exists():
        rng = np.random.default_rng(42 + hash(region) % 997)
        X = rng.normal(size=(60, N_FEATURES))
        y = (X[:, 0] * 1.2 + X[:, 1] * 0.8 + rng.normal(scale=0.5, size=60) > 0).astype(int)
        return X, y

    df = pd.read_csv(p, header=None)
    X = df.iloc[:, :N_FEATURES].values
    y = df.iloc[:, N_FEATURES].values.astype(int)

    uniq = np.unique(y)
    if len(uniq) < 2:
        rng = np.random.default_rng(123 + hash(region) % 1000)
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-3
        n_syn = max(12, max(6, X.shape[0] // 5))
        X_syn = rng.normal(loc=mu, scale=sigma, size=(n_syn, X.shape[1]))
        y_syn = np.array([1 - int(uniq[0])] * n_syn)
        X = np.vstack([X, X_syn]); y = np.concatenate([y, y_syn])
    return X, y

# ----------------- Model -----------------
def get_model():
    # class_weight="balanced" helps on small/imbalanced data
    m = LogisticRegression(solver="saga", max_iter=200, fit_intercept=True, class_weight="balanced")
    m.coef_ = np.zeros((1, N_FEATURES))
    m.intercept_ = np.zeros((1,))
    m.classes_ = np.array([0, 1])
    return m

def get_params_from_model(m: LogisticRegression):
    return [m.coef_.copy(), m.intercept_.copy()]

def set_model_params(m: LogisticRegression, params):
    m.coef_ = params[0].copy()
    m.intercept_ = params[1].copy()
    m.classes_ = np.array([0, 1])

def flatten_params(params):
    coef, inter = params
    return np.concatenate([coef.ravel(), inter.ravel()])

def unflatten_params(vec):
    coef = vec[:N_FEATURES].reshape(1, N_FEATURES)
    inter = vec[N_FEATURES:].reshape(1,)
    return [coef, inter]

def clip_by_l2_norm(v, max_norm):
    n = np.linalg.norm(v) + 1e-12
    if n <= max_norm: return v
    return v * (max_norm / n)

# ----------------- Privacy knobs -----------------
class PrivacyCfg:
    def __init__(self, use_dp=False, clip=1.0, dp_sigma=0.0, use_secagg=False, secagg_scale=0.0):
        self.use_dp = use_dp
        self.clip = clip
        self.dp_sigma = dp_sigma
        self.use_secagg = use_secagg
        self.secagg_scale = secagg_scale

# ----------------- Flower client -----------------
class RegionNumPyClient(fl.client.NumPyClient):
    def __init__(self, region: str, priv: PrivacyCfg, mask_vec: Optional[np.ndarray]):
        self.region = region
        self.priv = priv
        self.mask_vec = mask_vec  # for SecAgg-style masking (sums to zero globally)
        self.model = get_model()

    def get_parameters(self, config):
        return get_params_from_model(self.model)

    def fit(self, parameters, config):
        # Set model to global params
        set_model_params(self.model, parameters)
        old = flatten_params(get_params_from_model(self.model))

        # Local training
        X, y = load_xy(self.region)
        self.model.fit(X, y)
        new = flatten_params(get_params_from_model(self.model))
        delta = new - old  # update

        # --- Layer A: Local DP (clip + Gaussian noise) ---
        if self.priv.use_dp:
            delta = clip_by_l2_norm(delta, self.priv.clip)
            if self.priv.dp_sigma > 0:
                noise = np.random.normal(loc=0.0, scale=self.priv.dp_sigma, size=delta.shape)
                delta = delta + noise

        # --- Layer B: SecAgg-style masking (demo) ---
        # Each client adds a random mask; masks sum to zero across clients → aggregate un-masks
        if self.priv.use_secagg and self.mask_vec is not None:
            delta = delta + self.mask_vec

        # Return updated params (old + protected delta)
        protected_new = old + delta
        return unflatten_params(protected_new), len(X), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        X, y = load_xy(self.region)
        acc = self.model.score(X, y)
        loss = float(1.0 - acc)
        return loss, len(X), {"acc": acc}

# ----------------- Strategy that saves the aggregated model -----------------
class SavingFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggr, metrics = super().aggregate_fit(server_round, results, failures)
        if aggr is not None:
            self.latest_parameters = aggr
        # Immediately discard per-client params (demo: do not keep raw updates)
        return aggr, metrics

def export_joblib_from_parameters(parameters, out_joblib: Path):
    nds = parameters_to_ndarrays(parameters)
    mdl = get_model()
    set_model_params(mdl, nds)
    out_joblib.parent.mkdir(exist_ok=True)
    joblib.dump(mdl, out_joblib)

# ----------------- Build masks that sum to zero -----------------
def build_zero_sum_masks(num_clients: int, dim: int, scale: float, seed: int = 2024):
    if scale <= 0: return [None] * num_clients
    rng = np.random.default_rng(seed)
    masks = [rng.normal(0.0, scale, size=(dim,)) for _ in range(num_clients - 1)]
    last = -sum(masks)
    masks.append(last)
    return masks

# ----------------- Main -----------------
def main(rounds: int, use_dp: bool, clip: float, dp_sigma: float, use_secagg: bool, secagg_scale: float):
    # privacy config
    priv = PrivacyCfg(use_dp=use_dp, clip=clip, dp_sigma=dp_sigma, use_secagg=use_secagg, secagg_scale=secagg_scale)

    # Build clients
    base_clients = []
    dim = N_FEATURES + 1  # coef(3) + intercept(1)
    masks = build_zero_sum_masks(len(REGIONS), dim, secagg_scale) if use_secagg else [None]*len(REGIONS)
    for i, r in enumerate(REGIONS):
        base_clients.append(RegionNumPyClient(r, priv, masks[i]))

    # Convert to Client API if available
    clients = []
    for nc in base_clients:
        to_client = getattr(nc, "to_client", None)
        clients.append(to_client() if callable(to_client) else nc)

    # Choose legacy/new signature automatically
    sig = inspect.signature(fl.simulation.start_simulation)
    def client_fn_legacy(cid: str): return clients[int(cid) % len(clients)]
    def client_fn_context(context): return clients[int(getattr(context, "node_id", 0)) % len(clients)]
    use_legacy = not hasattr(fl.simulation, "ClientFnContext")

    strategy = SavingFedAvg()
    kwargs = {
        "num_clients": len(clients),
        "config": fl.server.ServerConfig(num_rounds=rounds),
        "strategy": strategy,
        "client_resources": {"num_cpus": 1},
    }
    kwargs["client_fn"] = client_fn_legacy if use_legacy else client_fn_context

    # Run simulation
    fl.simulation.start_simulation(**kwargs)

    # Save aggregated model
    out_joblib = Path("models") / "reranker.joblib"
    if strategy.latest_parameters is None:
        raise RuntimeError("No aggregated parameters were produced by the strategy.")
    export_joblib_from_parameters(strategy.latest_parameters, out_joblib)
    print(f"[OK] Saved global reranker to {out_joblib}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    # DP knobs
    ap.add_argument("--use-dp", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0, help="L2 clip bound for updates")
    ap.add_argument("--dp-sigma", type=float, default=0.0, help="Gaussian noise stddev added after clipping")
    # SecAgg-style knobs
    ap.add_argument("--use-secagg", action="store_true")
    ap.add_argument("--secagg-scale", type=float, default=0.0, help="Stddev of random masks; masks sum to zero")
    args = ap.parse_args()
    main(args.rounds, args.use_dp, args.clip, args.dp_sigma, args.use_secagg, args.secagg_scale)
