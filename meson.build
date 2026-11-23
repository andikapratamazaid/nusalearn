project(
  'nusalearn',
  'c', 'cpp', 'cython',
  version: '0.1.0',
  license: 'MIT',
  meson_version: '>=1.1',
  default_options: ['warning_level=1'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# NumPy include directory, needed by all three extension modules.
incdir_numpy = run_command(
  py, '-c',
  'import numpy, os; print(os.path.relpath(numpy.get_include()))',
  check: true,
).stdout().strip()
inc_np = include_directories(incdir_numpy)

subdir('nusalearn')
