This is the python module to use **SHTns, a high performance spherical harmonic transform and rotation** library.
It requires FFTW installed in your system.

## GETTING STARTED

### Basic Installation:
- make sure FFTW is installed in your system; if not, install it using your system package manager.
- run:

    `pip install shtns`

This simple installation should work and be enough for most use cases.
However, **in order to have the highest possible performance, FFTW should be compiled and tuned for your machine**,
which is not the case with pre-compiled system packages.

### Cuda GPU support:
Make sure the environment variable `CUDA_PATH` points to the installed cuda toolkit before running `pip install shtns`.
This will automatically add transparent support for nvidia gpus, in "offload" mode (that means data transfers are
performed by shtns).

### Example code:
See <https://bitbucket.org/nschaeff/shtns/src/master/examples/SHT_example.py>
and <https://bitbucket.org/nschaeff/shtns/src/master/examples/shallow_water.py>.
For using the GPU from python, see <https://bitbucket.org/nschaeff/shtns/src/master/examples/SHT_gpu_example.py>

## MORE INFO
See <https://bitbucket.org/nschaeff/shtns/src/master/README.md>
