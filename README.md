# NusaLearn

A small machine learning library written from scratch on top of NumPy,
with the fit / predict / score API style popularised by scikit-learn.

Every estimator in this repository is an original implementation
written for learning purposes. NusaLearn is not affiliated with
scikit-learn, does not use scikit-learn code, and does not depend on
scikit-learn at runtime. The only runtime dependency is NumPy.

## What is included

Estimators and tools, all in the `nusalearn` package:

- `linear_model.LinearRegression`: ordinary least squares, solved with
  the normal equations or batch gradient descent.
- `linear_model.LogisticRegression`: multinomial logistic regression
  trained by gradient descent, with `predict_proba`.
- `neighbors.KNeighborsClassifier` and `KNeighborsRegressor`: majority
  vote and neighbour averaging, with uniform or distance weights.
- `tree.DecisionTreeClassifier`: CART with Gini or entropy criteria,
  depth and leaf size controls.
- `cluster.KMeans`: k-means++ initialisation, multiple restarts,
  `labels_`, `cluster_centers_` and `inertia_`.
- `preprocessing.StandardScaler` and `MinMaxScaler`.
- `model_selection`: `train_test_split` (with stratification) and
  `cross_val_score` over k-fold splits.
- `metrics`: accuracy, confusion matrix, precision, recall and F1
  (binary and macro), ROC AUC (tie aware, rank based), mean squared
  error and R2.

## Architecture and language mix

Most of the library is Python with NumPy (about 1,070 lines in the
package). Three small compiled kernels sit underneath, each used by a
real code path:

- Cython (`nusalearn/_fast.pyx`, about 35 lines): pairwise Euclidean
  distances, used by k-nearest neighbors and KMeans.
- C (`nusalearn/_dot.c`, about 60 lines): a dot product used inside
  `LinearRegression.predict`.
- C++ (`nusalearn/_matops.cpp`, about 90 lines): a matrix multiply
  used to form X^T X in the normal equations solver.

The build system is Meson (about 115 lines of `meson.build` across the
repository) driven by meson-python, plus small `build.sh` and
`test.sh` shell scripts. Tests, examples and the benchmark script add
about 960 more lines of Python.

Every compiled kernel has a NumPy fallback, so the library works in a
pure Python checkout with no compiler at all. Setting the environment
variable `NUSALEARN_NO_KERNEL=1` forces the fallback path, which the
test suite uses to verify both paths agree.

## Installation

Requirements: Python 3.10 or newer and NumPy. Building the compiled
kernels additionally needs a C and C++ compiler, Meson, Ninja and
Cython, all installed automatically into the build environment by pip
build isolation.

The easy path uses the included script, which creates a virtual
environment in `.venv`, installs the toolchain, and installs the
project in editable mode:

```bash
./build.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install meson-python "meson>=1.1" ninja "Cython>=3.0" "numpy>=1.26" pytest
pip install --no-build-isolation -e .
```

If the compiled modules are missing for any reason, imports still
succeed and the NumPy fallbacks take over; only speed, not behaviour,
is affected (and, as the benchmark below shows, not even much speed).

## Quickstart

```python
import numpy as np
from nusalearn.neighbors import KNeighborsClassifier

X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
y = np.array(["a", "a", "b", "b"])

clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
print(clf.predict([[0.1, 0.2]]))   # ['a']
```

Running the bundled examples (from the repository root, in the
project environment):

```bash
python examples/iris_classification.py
python examples/kmeans_clustering.py
```

Measured output of the Iris example (raw features, stratified split,
seed 42, 105 train and 45 test samples):

```text
KNeighborsClassifier(k=5)
  test accuracy: 0.9556
DecisionTreeClassifier
  test accuracy: 0.9333
```

Measured output of the KMeans example (synthetic blobs, 300 samples):

```text
KMeans(n_clusters=3) converged in 4 iterations
  inertia (sum of squared distances to centroids): 410.53
  cluster sizes: [99, 101, 100]
  label match after best permutation: 0.9967
```

## Datasets

The `data` folder holds two small real datasets and two synthetic
ones. Full attribution is in `data/SOURCES.md`.

- `iris.csv`: 150 samples, 4 features, 3 classes. UCI Machine
  Learning Repository, Iris dataset. Fisher, R.A. (1936), "The use of
  multiple measurements in taxonomic problems", Annals of Eugenics
  7(2); data collected by Edgar Anderson. Values unchanged; a header
  row was added.
- `wine.csv`: 178 samples, 13 features, 3 classes. UCI Machine
  Learning Repository, Wine dataset; original data by Forina, M. et
  al., University of Genova. Values unchanged; a header row was added.
- `synthetic_regression.csv` and `synthetic_blobs.csv`: SYNTHETIC,
  generated locally by `data/make_synthetic.py` with seed 42. The
  regression target is a known linear combination plus Gaussian noise;
  the blobs are three Gaussians with known labels. Results on these
  files demonstrate correctness of the implementations only.

## Benchmark against scikit-learn

Measured with `benchmarks/benchmark_vs_sklearn.py` on the machine
that built this repository (Python 3.12, NumPy 2.x, scikit-learn
1.9.1). Raw numbers are in `benchmarks/results.csv` and the chart is
`charts/benchmark_accuracy.png`.

Methodology: identical stratified split for both libraries
(test_size 0.3, seed 42). kNN and logistic regression receive
StandardScaler features in both libraries; the decision tree receives
raw features. Logistic regression solvers differ by design (gradient
descent here, lbfgs in scikit-learn), so fit times are a snapshot, not
a tuned contest. Times are means over repeated runs after one warm-up.

| Dataset | Model | NusaLearn accuracy | scikit-learn accuracy | NusaLearn fit (ms) | scikit-learn fit (ms) |
| --- | --- | --- | --- | --- | --- |
| Iris | kNN (k=5) | 0.9333 | 0.9333 | 0.032 | 0.643 |
| Iris | Decision tree | 0.9333 | 0.9333 | 4.572 | 1.507 |
| Iris | Logistic regression | 0.9556 | 0.9556 | 113.908 | 7.362 |
| Wine | kNN (k=5) | 0.9091 | 0.9091 | 0.012 | 3.021 |
| Wine | Decision tree | 0.9091 | 0.9091 | 25.276 | 1.026 |
| Wine | Logistic regression | 0.9636 | 0.9636 | 121.900 | 7.689 |

Two honest observations:

- Accuracy matches scikit-learn exactly on all six jobs. On datasets
  this small that is the expected outcome for correct implementations
  of the same algorithms, and it is the main correctness signal here.
- NusaLearn loses on fit time where real training happens: the pure
  Python tree is about 3x slower on Iris and about 25x slower on Wine,
  and gradient descent logistic regression is about 15x slower than
  lbfgs. kNN "fit" is faster only because it stores the data and does
  no training at all.

### The Cython kernel, honestly

The kernel was expected to speed up distance computations. Measured
on a synthetic set of 20,000 training points with 16 features and
2,000 queries:

| Measurement | Kernel | NumPy fallback | Ratio |
| --- | --- | --- | --- |
| Pairwise distances only | 418.9 ms | 417.0 ms | 1.00x |
| Full kNN predict | 6,652.2 ms | 6,338.9 ms | 0.95x |

There is no measurable speedup. The NumPy fallback computes distances
through the Gram matrix identity, which delegates to an optimised
BLAS matrix multiply, and that is already as fast as the
straightforward Cython loops. End-to-end predict time is dominated by
sorting 2,000 by 20,000 distances and by the Python voting loop, which
the kernel does not touch. The kernel stays in the project because it
is numerically correct and demonstrates the build path, but nobody
should cite it as a performance feature.

## Tests

46 tests cover every estimator and metric: hand-checkable cases
(exact line recovery, a 3-4-5 distance triangle, known ROC AUC
values), dataset level accuracy thresholds, kernel versus NumPy
agreement, and data splitting behaviour. Run them with `./test.sh` or
directly:

```bash
python -m pytest tests -q                              # 46 passed
NUSALEARN_NO_KERNEL=1 python -m pytest tests -q        # 46 passed
```

Both modes pass 46 of 46: the compiled path and the pure Python
fallback behave the same. Writing the suite caught two real bugs,
both fixed: the kNN classifier crashed on plain list input to
`predict`, and the NumPy distance path left tiny floating point
residue on the self-distance diagonal.

## Limitations

- Educational scope. No sparse matrices, no sample weights, no
  pipelines, no probability calibration, no parallelism.
- Greedy CART splitting cannot start on problems like XOR, where the
  first split gives zero impurity decrease; the tree then stays a
  single leaf. A test pins this behaviour.
- Logistic regression is plain batch gradient descent: it needs
  scaled features and patience, and it is much slower than lbfgs.
- kNN predict sorts all distances in Python and votes in a Python
  loop, so it does not scale to large training sets.
- The compiled kernels match, but do not beat, the NumPy fallbacks on
  the workloads measured here.
- Only the six estimators above exist. This is a learning project,
  not a replacement for scikit-learn.

## License

MIT. See `LICENSE`. The Iris and Wine datasets remain the work of
their original authors and are distributed by the UCI Machine
Learning Repository for research and education use; see
`data/SOURCES.md` for attribution.
