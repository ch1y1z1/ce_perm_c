# ceviche package build configuration

# Dependencies
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

py.install_sources(
  ['__init__.py', 'constants.py', 'derivatives.py', 'fdfd.py', 'fdtd.py', 'jacobians.py', 'modes.py', 'optimizers.py', 'primitives.py', 'solvers.py', 'sources.py', 'utils.py', 'viz.py'],
  subdir: 'ce_perm_c/ceviche'
)
