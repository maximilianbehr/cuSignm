/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include "checkcuda.h"
#include "cusignm.h"
#include "cusignm_frobenius.h"
#include "cusignm_traits.h"

const static cusolverAlgMode_t CUSOLVER_ALG = CUSOLVER_ALG_0;

template <typename T>
static int cusignm_NewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    /*-----------------------------------------------------------------------------
     * initialize with zero
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = 0;
    *h_bufferSize = 0;

    /*-----------------------------------------------------------------------------
     * get device and host workspace size for LU factorization
     *-----------------------------------------------------------------------------*/
    // create cusolver handle
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // create cusolver params
    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    // compute workspace size
    CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cusignm_traits<T>::dataType, nullptr, n, cusignm_traits<T>::computeType, d_bufferSize, h_bufferSize));

    // free workspace
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));

    /*-----------------------------------------------------------------------------
     * compute final workspace size
     *-----------------------------------------------------------------------------*/
    *d_bufferSize += sizeof(T) * n * n * 3 + sizeof(int64_t) * n + sizeof(int);

    return 0;
}

int cusignm_sNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cusignm_NewtonBufferSize<float>(n, d_bufferSize, h_bufferSize);
}

int cusignm_dNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cusignm_NewtonBufferSize<double>(n, d_bufferSize, h_bufferSize);
}

int cusignm_cNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cusignm_NewtonBufferSize<cuComplex>(n, d_bufferSize, h_bufferSize);
}

int cusignm_zNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cusignm_NewtonBufferSize<cuDoubleComplex>(n, d_bufferSize, h_bufferSize);
}

template <typename T>
__global__ void identity(const int n, T *A, const int lda) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    for (int j = j0; j < n; j += gridDim.y * blockDim.y) {
        for (int i = i0; i < n; i += gridDim.x * blockDim.x) {
            A[i + j * lda] = (i == j) ? cusignm_traits<T>::one : cusignm_traits<T>::zero;
        }
    }
}

template <typename T>
static int cusignm_Newton(const int n, const T *A, void *d_buffer, void *h_buffer, T *S) {
    /*-----------------------------------------------------------------------------
     * derived types
     *-----------------------------------------------------------------------------*/
    using Scalar = typename cusignm_traits<T>::S;  // real type: double for cuDoubleComplex, float for cuComplex

    /*-----------------------------------------------------------------------------
     * constants and variables
     *-----------------------------------------------------------------------------*/
    int ret = 0, iter = 1;
    constexpr int maxiter = 100;
    const Scalar tol = std::sqrt(std::numeric_limits<Scalar>::epsilon());  // square root of machine epsilon - newton iteration converges quadratically
    Scalar alpha, beta, mu;

    /*-----------------------------------------------------------------------------
     * create cuBlas handle
     *-----------------------------------------------------------------------------*/
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * create cusolver handle and params structure
     *-----------------------------------------------------------------------------*/
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    /*-----------------------------------------------------------------------------
     * split memory buffer
     * memory layout: |Sold, Stmp, SoldInv, ipiv, info, d_work|
     *-----------------------------------------------------------------------------*/
    T *Sold = reinterpret_cast<T *>(d_buffer);
    T *Stmp = reinterpret_cast<T *>(Sold + n * n);                   // put Stmp after Sold
    T *SoldInv = reinterpret_cast<T *>(Stmp + n * n);                // put SoldInv after Sold
    int64_t *d_ipiv = reinterpret_cast<int64_t *>(SoldInv + n * n);  // put d_ipiv after SoldInv
    int *d_info = reinterpret_cast<int *>(d_ipiv + n);               // put d_info after d_ipiv
    void *d_work = reinterpret_cast<int *>(d_info + 1);              // put d_work after d_info
    void *h_work = reinterpret_cast<void *>(h_buffer);
    std::swap(S, Sold);

    /*-----------------------------------------------------------------------------
     * copy A to Sold
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaMemcpy(Sold, A, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));

    /*-----------------------------------------------------------------------------
     * newton iteration
     *-----------------------------------------------------------------------------*/
    iter = 1;
    static_assert(maxiter >= 1, "maxiter >= 1");
    while (true) {
        /*-----------------------------------------------------------------------------
         * copy Sold to Stmp
         *-----------------------------------------------------------------------------*/
        CHECK_CUDA(cudaMemcpy(Stmp, Sold, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));

        /*-----------------------------------------------------------------------------
         * compute inv(S)^H
         *-----------------------------------------------------------------------------*/
        // workspace query for LU factorization
        size_t lworkdevice = 0, lworkhost = 0;
        CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cusignm_traits<T>::dataType, Stmp, n, cusignm_traits<T>::computeType, &lworkdevice, &lworkhost));

        // compute LU factorization and set right side to identity on different streams
        cudaStream_t streamLU, streamIdentity;
        CHECK_CUDA(cudaStreamCreate(&streamLU));
        CHECK_CUDA(cudaStreamCreate(&streamIdentity));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, streamLU));
        CHECK_CUSOLVER(cusolverDnXgetrf(cusolverH, params, n, n, cusignm_traits<T>::dataType, Stmp, n, d_ipiv, cusignm_traits<T>::computeType, d_work, lworkdevice, h_work, lworkhost, d_info));

        // set right-hand side to identity
        {
            dim3 grid((n + 15) / 16, (n + 15) / 16);
            dim3 block(16, 16);
            identity<<<grid, block, 0, streamIdentity>>>(n, SoldInv, n);
            CHECK_CUDA(cudaPeekAtLastError());
        }

        // synchronize and destroy streams
        CHECK_CUDA(cudaStreamSynchronize(streamLU));
        CHECK_CUDA(cudaStreamSynchronize(streamIdentity));
        CHECK_CUDA(cudaStreamDestroy(streamLU));
        CHECK_CUDA(cudaStreamDestroy(streamIdentity));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, 0));

        // solve the linear system to compute the hermitian/transposed inverse of Sold
        CHECK_CUSOLVER(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n, cusignm_traits<T>::dataType, Stmp, n, d_ipiv, cusignm_traits<T>::computeType, SoldInv, n, d_info));

        /*-----------------------------------------------------------------------------
         * compute alpha and beta to compute mu for the first iteration
         *-----------------------------------------------------------------------------*/
        if (iter == 1) {
            CHECK_CUSIGNM(cusignm_normFro(n, n, A, &alpha));
            CHECK_CUSIGNM(cusignm_normFro(n, n, SoldInv, &beta));
            beta = Scalar{1.0} / beta;
            mu = Scalar{1.0} / std::sqrt(alpha * beta);
        }

        /*-----------------------------------------------------------------------------
         * update S as 0.5 * (mu * S + 1/mu * inv(S))
         *-----------------------------------------------------------------------------*/
        {
            T a, b;
            if constexpr (std::is_same<T, cuComplex>::value || std::is_same<T, cuDoubleComplex>::value) {
                a = T{Scalar{0.5} * mu, Scalar{0.0}}, b = T{Scalar{0.5} / mu, Scalar{0.0}};
            } else {
                a = Scalar{0.5} * mu, b = Scalar{0.5} / mu;
            }
            CHECK_CUBLAS(cusignm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &a, Sold, n, &b, SoldInv, n, S, n));
        }

        /*-----------------------------------------------------------------------------
         * update mu for the next iteration
         *-----------------------------------------------------------------------------*/
        if (iter == 1) {
            mu = std::sqrt(Scalar{2.0} * std::sqrt(alpha * beta) / (alpha + beta));
        } else {
            mu = std::sqrt(Scalar{2.0} / (mu + Scalar{1.0} / mu));
        }

        /*-----------------------------------------------------------------------------
         *  compute relative change of S and Sold
         *-----------------------------------------------------------------------------*/
        Scalar diffSSold, nrmS;
        CHECK_CUSIGNM(cusignm_diffnormFro(n, n, S, Sold, &diffSSold));
        CHECK_CUSIGNM(cusignm_normFro(n, n, S, &nrmS));
        // printf("iter=%d, diffSSold=%e, nrmS=%e, rel. change=%e\n", iter, diffSSold, nrmS, diffSSold / nrmS);

        /*-----------------------------------------------------------------------------
         * stopping criteria
         *-----------------------------------------------------------------------------*/
        // relative change of S and Sold is smaller than tolerance
        if (diffSSold < nrmS * tol) {
            break;
        }

        if (isnan(diffSSold) || isnan(nrmS)) {
            fprintf(stderr, "%s-%s:%d no convergence - NaN detected\n", __func__, __FILE__, __LINE__);
            fflush(stderr);
            ret = -1;
            break;
        }

        // maximum number of iterations reached
        if (iter == maxiter) {
            fprintf(stderr, "%s-%s:%d no convergence - maximum number of iterations reached\n", __func__, __FILE__, __LINE__);
            fflush(stderr);
            ret = -1;
            break;
        }

        /*-----------------------------------------------------------------------------
         * swap S and Sold for the next iteration
         *-----------------------------------------------------------------------------*/
        std::swap(S, Sold);
        iter++;
    }

    /*-----------------------------------------------------------------------------
     * copy S and Sold if necessary
     *-----------------------------------------------------------------------------*/
    if (iter % 2 == 1) {
        CHECK_CUDA(cudaMemcpy(Sold, S, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));
    }

    /*-----------------------------------------------------------------------------
     * destroy cuBlas and cuSolver handle and params structure
     *-----------------------------------------------------------------------------*/
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return ret;
}

int cusignm_sNewton(const int n, const float *A, void *d_buffer, void *h_buffer, float *S) {
    return cusignm_Newton(n, A, d_buffer, h_buffer, S);
}

int cusignm_dNewton(const int n, const double *A, void *d_buffer, void *h_buffer, double *S) {
    return cusignm_Newton(n, A, d_buffer, h_buffer, S);
}

int cusignm_cNewton(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *S) {
    return cusignm_Newton(n, A, d_buffer, h_buffer, S);
}

int cusignm_zNewton(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *S) {
    return cusignm_Newton(n, A, d_buffer, h_buffer, S);
}
