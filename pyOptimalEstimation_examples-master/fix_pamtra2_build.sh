#!/bin/bash
export LDFLAGS="-Wl,-undefined,dynamic_lookup"
export FFLAGS="-Wl,-undefined,dynamic_lookup"
export NPY_DISTUTILS_APPEND_FLAGS=0
export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"

# Ensure we're using the right compilers
export FC=/opt/homebrew/bin/gfortran
export F77=/opt/homebrew/bin/gfortran
export F90=/opt/homebrew/bin/gfortran
export CC=clang
export CXX=clang++

conda env create -f environment_macosx_original.yml