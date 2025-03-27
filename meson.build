project('ce_perm_c', 'c',
  version: '0.1.0',
  default_options: [
    'warning_level=1',
    'buildtype=debugoptimized',
  ],
)

# Configure Python module
py_mod = import('python')
py = py_mod.find_installation(pure: false)

# Add the ce_perm_c subdirectory
subdir('ce_perm_c')
