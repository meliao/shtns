# Python setup, with configure and make steps
# See: https://docs.python.org/3/distutils/apiref.html#module-distutils.command
# and https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext, new_compiler, customize_compiler
from numpy import get_include
import os,sys

def getver():
    with open('CHANGELOG.md') as f:
        for l in f:
            s=l.find('* v')
            if s>=0:
                return l[s+3:].split()[0]
    return 'unknown'

OPENMP_TEST_C = """
#include <omp.h>
#include "fftw3/fftw3.h"
int main(void) {
  fftw_init_threads();
  return omp_get_max_threads();
}
"""

def check_openmp_support(omp_flags='-fopenmp'):
    """ check if openmp is actually supported (not always on macos!) -- adapted from https://github.com/astropy/extension-helpers/blob/main/extension_helpers/_openmp_helpers.py """
    import tempfile,glob
    ccompiler = new_compiler()
    customize_compiler(ccompiler)
    openmp_ok = True
    with tempfile.TemporaryDirectory() as tmp_dir:
        start_dir = os.path.abspath('.')
        try:
            os.chdir(tmp_dir)
            # Write source of test program (will also test fftw openmp support!)
            with open('test_openmp.c', 'w') as f:
                f.write(OPENMP_TEST_C)
            os.mkdir('objects')
            # Compile test program
            ccompiler.compile(['test_openmp.c'], output_dir='objects', extra_postargs=[omp_flags, "-I"+start_dir])
            # Link test program with fftw3_omp library
            objects = glob.glob(os.path.join('objects', '*' + ccompiler.obj_extension))
            ccompiler.link_executable(objects, 'test_openmp', extra_postargs=[omp_flags, "-lfftw3_omp"])
        except Exception:
            openmp_ok = False
        finally:
            os.chdir(start_dir)
    return openmp_ok

numpy_inc = get_include()               #  NumPy include path.
shtns_o = "sht_init.o sht_kernels_a.o sht_kernels_s.o sht_odd_nlat.o sht_fly.o sht_omp.o".split()
libdir = []
cargs = ['-std=c99', '-DSHTNS_VER="' + getver() +'"']
libs = ['fftw3', 'm']
config_cmd = ['./configure','--enable-python','--prefix='+sys.prefix]

use_openmp = os.environ.get('SHTNS_OPENMP', '1') != '0'   # allows to disable openmp with environment variable SHTNS_OPENMP=0
if use_openmp:
    use_openmp = check_openmp_support()
if use_openmp:
    cargs.append('-fopenmp')
    libs.insert(0,'fftw3_omp')
else:
    config_cmd.append('--disable-openmp')

class make(build_ext):
    def run(self):
        self.spawn(config_cmd)
        self.spawn(['make','--jobs=4', *shtns_o])   # make the objects required to build extension
        super().run()

shtns_module = Extension('_shtns', sources=['shtns_numpy_wrap.c'],
        extra_objects=shtns_o, depends=shtns_o,
        extra_compile_args=cargs,
        library_dirs=libdir,
        libraries=libs,
        include_dirs=[numpy_inc])

setup(name='shtns',
    cmdclass={'build_ext': make },
        version=getver(),
        description='High performance Spherical Harmonic Transform',
        author='Nathanael Schaeffer',
        author_email='nathanael.schaeffer@univ-grenoble-alpes.fr',
        url='https://bitbucket.org/nschaeff/shtns',
        ext_modules=[shtns_module],
        py_modules=["shtns"],
        requires=["numpy"],
        )
