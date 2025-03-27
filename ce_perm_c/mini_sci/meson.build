# mini_sci package build configuration

# Dependencies
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Get numpy include directory
numpy_inc_dirs = run_command(py,
  ['-c', 'import os; os.chdir(os.path.join(os.environ.get("MESON_SOURCE_ROOT", "."), "ce_perm_c/mini_sci"));' +
         'import numpy; print(os.path.relpath(numpy.get_include()))'],
  check: true
).stdout().strip()

# Create numpy dependency
numpy_dep = declare_dependency(
  include_directories: include_directories(numpy_inc_dirs)
)

# Check for LAPACK/BLAS
cc = meson.get_compiler('c')
lapack_deps = []
blas_found = false

# Try to find LAPACK/BLAS
foreach libblas : ['blas', 'openblas', 'cblas']
  if not blas_found
    blas_dep = cc.find_library(libblas, required: false)
    if blas_dep.found()
      lapack_deps += blas_dep
      blas_found = true
    endif
  endif
endforeach

if not blas_found
  # Fallback if BLAS not found
  message('BLAS not found, trying to find it without specifying path')
  lapack_deps += cc.find_library('cblas', required: false)
  if not lapack_deps[0].found()
    message('CBLAS not found either, your build may fail')
  endif
endif

# SuperLU source directory
superlu_src_dir = 'SuperLU/SRC'

# Get all SuperLU source files
superlu_sources = run_command('find', superlu_src_dir, '-name', '*.c', check: true).stdout().strip().split('\n')

# Define SuperLU macros
superlu_defs = ['-DUSE_VENDOR_BLAS=1']

# Create SuperLU library
superlu_lib = static_library('superlu_src',
  superlu_sources,
  c_args: superlu_defs,
  include_directories: include_directories(superlu_src_dir),
  dependencies: lapack_deps,
)

# Extension module sources
ext_sources = [
  '_superlumodule.c',
  '_superlu_utils.c',
  '_superluobject.c',
]

# Define numpy API macros
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']

# Create Python extension module
py.extension_module('_superlu',
  ext_sources,
  link_with: superlu_lib,
  dependencies: [py_dep, numpy_dep, lapack_deps],
  c_args: numpy_nodepr_api,
  include_directories: [include_directories(superlu_src_dir), include_directories(numpy_inc_dirs)],
  install: true,
)

# Install Python files
py.install_sources(
  ['linsolve.py'],
  subdir: 'ce_perm_c/mini_sci'
)
