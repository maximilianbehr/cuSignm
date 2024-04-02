# cuSignm - Matrix Sign Function Approximation using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuSignm)
 [![DOI](https://zenodo.org/badge/780356669.svg)](https://zenodo.org/doi/10.5281/zenodo.10908661)

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuSignm` is a `CUDA` library implementing Newton and Halley's method for the Sign Matrix Function Approximation of a square matrix $A$ with no eigenvalues on the imaginary axis.

`cuSignm` supports real and complex, single and double precision matrices.

## Available Functions


### Single Precision Functions
```C
int cusignm_sNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_sNewton(const int n, const float *A, void *d_buffer, void *h_buffer, float *S);

int cusignm_sHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_sHalley(const int n, const float *A, void *d_buffer, void *h_buffer, float *S);
```

### Double Precision Functions
```C
int cusignm_dNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_dNewton(const int n, const double *A, void *d_buffer, void *h_buffer, double *S);

int cusignm_dHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_dHalley(const int n, const double *A, void *d_buffer, void *h_buffer, double *S);
```

### Complex Single Precision Functions
```C
int cusignm_cNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_cNewton(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *S);

int cusignm_cHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_cHalley(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *S);
```

### Complex Double Precision Functions
```C
int cusignm_zNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_zNewton(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *S);

int cusignm_zHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_zHalley(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *S);
```


## Algorithm

`cuSignm` implements the Newton's method with scaling as well as Halley's method with scaling for the approximation of the matrix sign function.
The matrix $A$ must be square and with no eigenvalues on the imaginary axis.

See also Algorithm 3.1
> Chen, J., & Chow, E. (2014). A stable scaling of Newton-Schulz for improving the sign function computation of a Hermitian matrix. _Preprint_. ANL/MCS-P5059-0114.


## Installation

Prerequisites:
 * `CMake >= 3.23`
 * `CUDA >= 11.4.2`

```shell
  mkdir build && cd build
  cmake ..
  make
  make install
```

## Usage and Examples

We provide examples for all supported matrix formats:
  
| File                                                       | Data                                |
| -----------------------------------------------------------|-------------------------------------|
| [`example_cusignm_sNewton.cu`](example_cusignm_sNewton.cu) | real, single precision matrix       |
| [`example_cusignm_dNewton.cu`](example_cusignm_dNewton.cu) | real, double precision matrix       |
| [`example_cusignm_cNewton.cu`](example_cusignm_cNewton.cu) | complex, single precision matrix    |
| [`example_cusignm_zNewton.cu`](example_cusignm_zNewton.cu) | complex, double precision matrix    |
| [`example_cusignm_sHalley.cu`](example_cusignm_sHalley.cu) | real, single precision matrix       |
| [`example_cusignm_dHalley.cu`](example_cusignm_dHalley.cu) | real, double precision matrix       |
| [`example_cusignm_cHalley.cu`](example_cusignm_cHalley.cu) | complex, single precision matrix    |
| [`example_cusignm_zHalley.cu`](example_cusignm_zHalley.cu) | complex, double precision matrix    |
