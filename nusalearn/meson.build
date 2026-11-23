# Compiled extension modules. All three are optional at runtime: the
# Python code falls back to NumPy when a module is missing.

py.extension_module(
  '_fast',
  '_fast.pyx',
  dependencies: py_dep,
  include_directories: inc_np,
  install: true,
  subdir: 'nusalearn',
)

py.extension_module(
  '_dot',
  '_dot.c',
  dependencies: py_dep,
  include_directories: inc_np,
  install: true,
  subdir: 'nusalearn',
)

py.extension_module(
  '_matops',
  '_matops.cpp',
  dependencies: py_dep,
  include_directories: inc_np,
  install: true,
  subdir: 'nusalearn',
)

python_sources = [
  '__init__.py',
  '_native.py',
  '_pairwise.py',
  'base.py',
  'cluster.py',
  'linear_model.py',
  'metrics.py',
  'model_selection.py',
  'neighbors.py',
  'preprocessing.py',
  'tree.py',
  'validation.py',
]

py.install_sources(python_sources, subdir: 'nusalearn')
