# JAX implementation notes


| Transform | JAX CPU | JAX CUDA | JAX autodiff (CPU) | JAX autodiff (CUDA) | Notes |
| --- | --- | --- | --- | --- | --- |
| synth | Y | Y | Y | Y | |
| analys | Y | Y | Y | Y | |
| synth_cplx | Y | N | Y | N | Native CUDA not implemented. |
| analys_cplx | Y | N | Y | N | Native CUDA not implemented. |
| vector synth | N | N | N | N | |
| vector analys | N | N | N | N | |
| cplx vector synth | N | N | N | N | |
| cplx vector analys | N | N | N | N | |