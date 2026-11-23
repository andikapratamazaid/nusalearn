"""Decision tree: hand cases and dataset checks."""

import numpy as np

from nusalearn.metrics import accuracy_score
from nusalearn.model_selection import train_test_split
from nusalearn.tree import DecisionTreeClassifier


def test_tree_learns_simple_threshold():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0
    assert tree.predict([[0.5]])[0] == 0
    assert tree.predict([[2.5]])[0] == 1


def test_tree_learns_and_function():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0


def test_tree_greedy_splitting_cannot_start_xor():
    # Known limitation of greedy impurity based splitting: the first
    # split of XOR gives zero impurity decrease on either feature, so
    # the tree stays a single leaf. This test pins that behaviour.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    tree = DecisionTreeClassifier().fit(X, y)
    assert accuracy_score(y, tree.predict(X)) < 1.0


def test_tree_entropy_criterion():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(criterion="entropy").fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0


def test_tree_max_depth_one_is_a_stump():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    # A stump cannot represent XOR, so training accuracy is below 1.
    assert accuracy_score(y, stump.predict(X)) < 1.0


def test_tree_on_iris(iris):
    X, y = iris
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=True)
    tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
    assert accuracy_score(y_te, tree.predict(X_te)) >= 0.90
