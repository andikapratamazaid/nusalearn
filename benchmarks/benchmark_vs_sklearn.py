"""Benchmark NusaLearn against scikit-learn on Iris and Wine.

Run from the repository root, in the project environment:

    python benchmarks/benchmark_vs_sklearn.py

Methodology, kept deliberately simple and identical for both
libraries:

- Same stratified split for everyone (test_size=0.3, seed 42).
- kNN and logistic regression get StandardScaler features in both
  libraries (distance based and gradient based methods need comparable
  feature scales; the logistic solvers differ, gradient descent here
  versus lbfgs in scikit-learn, so treat the timing as a snapshot,
  not a tuned contest).
- The decision tree uses raw features in both libraries.
- Fit and predict times are means over repeated runs after one
  warm-up run, measured with time.perf_counter.
- The kNN speed test times full predict calls on a larger synthetic
  set (20,000 train x 16 features, 2,000 queries), once with the
  compiled Cython kernel and once with NUSALEARN_NO_KERNEL=1 forcing
  the pure NumPy fallback, each in a fresh subprocess so the import
  time environment is honest.

Outputs: benchmarks/results.csv and charts/benchmark_accuracy.png.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nusalearn.linear_model import LogisticRegression as NLLogReg  # noqa: E402
from nusalearn.metrics import accuracy_score as nl_accuracy  # noqa: E402
from nusalearn.model_selection import train_test_split as nl_split  # noqa: E402
from nusalearn.neighbors import KNeighborsClassifier as NLKNN  # noqa: E402
from nusalearn.preprocessing import StandardScaler as NLScaler  # noqa: E402
from nusalearn.tree import DecisionTreeClassifier as NLTree  # noqa: E402

from sklearn.linear_model import LogisticRegression as SKLogReg  # noqa: E402
from sklearn.metrics import accuracy_score as sk_accuracy  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier as SKKNN  # noqa: E402
from sklearn.preprocessing import StandardScaler as SKScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier as SKTree  # noqa: E402


def load_csv(name):
    raw = np.genfromtxt(ROOT / "data" / name, delimiter=",", names=True,
                        dtype=None, encoding=None)
    return raw


def load_iris():
    raw = load_csv("iris.csv")
    X = np.column_stack([raw[n] for n in raw.dtype.names[:-1]]).astype(float)
    y = np.asarray(raw[raw.dtype.names[-1]])
    return X, y


def load_wine():
    raw = load_csv("wine.csv")
    names = raw.dtype.names
    y = np.asarray(raw[names[0]]).astype(int)
    X = np.column_stack([raw[n] for n in names[1:]]).astype(float)
    return X, y


def time_call(fn, repeats):
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0  # ms


def bench_dataset(name, X, y, fit_repeats):
    rows = []
    X_tr, X_te, y_tr, y_te = nl_split(X, y, test_size=0.3, random_state=42, stratify=True)
    scaler = NLScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    jobs = [
        # (model label, nusalearn factory, sklearn factory, scaled?)
        ("knn_k5",
         lambda: NLKNN(n_neighbors=5), lambda: SKKNN(n_neighbors=5), True),
        ("decision_tree",
         lambda: NLTree(random_state=42), lambda: SKTree(random_state=42), False),
        ("logistic_regression",
         lambda: NLLogReg(learning_rate=0.1, max_iter=2000),
         lambda: SKLogReg(max_iter=1000), True),
    ]
    for model_name, nl_factory, sk_factory, scaled in jobs:
        a_tr, a_te = (X_tr_s, X_te_s) if scaled else (X_tr, X_te)
        for library, factory, acc_fn in (
            ("nusalearn", nl_factory, nl_accuracy),
            ("sklearn", sk_factory, sk_accuracy),
        ):
            model = factory()
            fit_ms = time_call(lambda: factory().fit(a_tr, y_tr), fit_repeats)
            model.fit(a_tr, y_tr)
            pred_ms = time_call(lambda: model.predict(a_te), 30)
            acc = acc_fn(y_te, model.predict(a_te))
            rows.append({
                "dataset": name, "task": "classification", "model": model_name,
                "library": library, "accuracy": round(float(acc), 4),
                "fit_time_ms": round(fit_ms, 3), "predict_time_ms": round(pred_ms, 3),
            })
            print(f"{name:6s} {model_name:20s} {library:10s} "
                  f"acc={acc:.4f} fit={fit_ms:.3f} ms predict={pred_ms:.3f} ms")
    return rows


SPEED_SNIPPET = """
import time
import numpy as np
from nusalearn.neighbors import KNeighborsClassifier
from nusalearn._pairwise import KERNEL_AVAILABLE
rng = np.random.default_rng(123)
X_train = rng.normal(size=(20000, 16))
y_train = rng.integers(0, 3, size=20000)
X_query = rng.normal(size=(2000, 16))
clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
clf.predict(X_query[:10])
start = time.perf_counter()
clf.predict(X_query)
elapsed = (time.perf_counter() - start) * 1000.0
print(f"{elapsed:.3f} {KERNEL_AVAILABLE}")
"""


def bench_distance_kernel():
    """Time the distance computation itself, kernel versus NumPy.

    This isolates what the Cython kernel accelerates. End-to-end kNN
    predict also pays for argsort and the Python voting loop, which is
    why its speedup (measured separately) is smaller.
    """
    from nusalearn._pairwise import pairwise_distances, pairwise_distances_numpy

    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(20000, 16))
    X_query = rng.normal(size=(2000, 16))
    rows = []
    times = {}
    for mode, fn in (("kernel", pairwise_distances),
                     ("numpy", pairwise_distances_numpy)):
        ms = time_call(lambda: fn(X_query, X_train), 3)
        times[mode] = ms
        print(f"pairwise distances ({mode}): {ms:.1f} ms")
        rows.append({
            "dataset": "synthetic_20k_x_16", "task": "pairwise_distances",
            "model": mode, "library": "nusalearn",
            "accuracy": "", "fit_time_ms": "",
            "predict_time_ms": round(ms, 1),
        })
    speedup = times["numpy"] / times["kernel"]
    print(f"Cython kernel speedup for pairwise distances: {speedup:.2f}x")
    rows.append({
        "dataset": "synthetic_20k_x_16", "task": "pairwise_distances_speedup",
        "model": "kernel_vs_numpy", "library": "nusalearn",
        "accuracy": "", "fit_time_ms": "",
        "predict_time_ms": round(speedup, 2),
    })
    return rows


def bench_knn_speed():
    rows = []
    results = {}
    for mode, env_extra in (("kernel", {}), ("fallback", {"NUSALEARN_NO_KERNEL": "1"})):
        env = dict(os.environ)
        env.update(env_extra)
        out = subprocess.run(
            [sys.executable, "-c", SPEED_SNIPPET],
            capture_output=True, text=True, env=env, cwd=ROOT, check=True,
        )
        ms_str, kernel_str = out.stdout.strip().split()
        results[mode] = float(ms_str)
        print(f"knn predict ({mode}, kernel_available={kernel_str}): {float(ms_str):.1f} ms")
        rows.append({
            "dataset": "synthetic_20k_x_16", "task": "knn_predict_speed",
            "model": f"knn_k5_{mode}", "library": "nusalearn",
            "accuracy": "", "fit_time_ms": "",
            "predict_time_ms": round(float(ms_str), 1),
        })
    speedup = results["fallback"] / results["kernel"]
    print(f"Cython kernel speedup for kNN predict: {speedup:.2f}x")
    rows.append({
        "dataset": "synthetic_20k_x_16", "task": "knn_predict_speedup",
        "model": "kernel_vs_fallback", "library": "nusalearn",
        "accuracy": "", "fit_time_ms": "",
        "predict_time_ms": round(speedup, 2),
    })
    return rows


def make_chart(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls = df[df["task"] == "classification"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nl_rows = cls[cls.library == "nusalearn"]
    labels = [f"{r.dataset}\n{r.model}" for r in nl_rows.itertuples()]
    nl = cls[cls.library == "nusalearn"]["accuracy"].to_numpy()
    sk = cls[cls.library == "sklearn"]["accuracy"].to_numpy()
    x = np.arange(len(nl))
    axes[0].bar(x - 0.2, nl, width=0.4, label="NusaLearn")
    axes[0].bar(x + 0.2, sk, width=0.4, label="scikit-learn")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylim(0.8, 1.02)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Accuracy, Iris and Wine (same split, seed 42)")
    axes[0].legend()

    nl_t = cls[cls.library == "nusalearn"]["fit_time_ms"].to_numpy()
    sk_t = cls[cls.library == "sklearn"]["fit_time_ms"].to_numpy()
    axes[1].bar(x - 0.2, nl_t, width=0.4, label="NusaLearn")
    axes[1].bar(x + 0.2, sk_t, width=0.4, label="scikit-learn")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("mean fit time (ms, log scale)")
    axes[1].set_title("Fit time (lower is faster)")
    axes[1].legend()

    fig.tight_layout()
    out = ROOT / "charts" / "benchmark_accuracy.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"chart written to {out}")


def main():
    rows = []
    X, y = load_iris()
    rows += bench_dataset("iris", X, y, fit_repeats=50)
    X, y = load_wine()
    rows += bench_dataset("wine", X, y, fit_repeats=30)
    rows += bench_distance_kernel()
    rows += bench_knn_speed()
    df = pd.DataFrame(rows)
    out_csv = ROOT / "benchmarks" / "results.csv"
    df.to_csv(out_csv, index=False)
    print(f"results written to {out_csv}")
    make_chart(df)


if __name__ == "__main__":
    main()
