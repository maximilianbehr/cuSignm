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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>

#include "cusignm.h"

int main(void) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    int ret = 0;            // return value
    const int n = 10;       // size of the input matrix A n-by-n
    cuComplex *A, *S;       // input matrix and sign matrix
    void *d_buffer = NULL;  // device buffer
    void *h_buffer = NULL;  // host buffer

    /*-----------------------------------------------------------------------------
     * allocate A, Q and H
     *-----------------------------------------------------------------------------*/
    cudaMallocManaged((void **)&A, sizeof(*A) * n * n);
    cudaMallocManaged((void **)&S, sizeof(*S) * n * n);

    /*-----------------------------------------------------------------------------
     * create a random matrix A
     *-----------------------------------------------------------------------------*/
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i + j * n] = cuComplex({(float)rand() / RAND_MAX, (float)rand() / RAND_MAX});
        }
    }

    /*-----------------------------------------------------------------------------
     * perform a workspace query and allocate memory buffer on the host and device
     *-----------------------------------------------------------------------------*/
    size_t d_bufferSize = 0, h_bufferSize = 0;
    cusignm_cHalleyBufferSize(n, &d_bufferSize, &h_bufferSize);

    if (d_bufferSize > 0) {
        cudaMalloc((void **)&d_buffer, d_bufferSize);
    }

    if (h_bufferSize > 0) {
        h_buffer = malloc(h_bufferSize);
    }

    /*-----------------------------------------------------------------------------
     *  compute Sign Function S = sign(A)
     *-----------------------------------------------------------------------------*/
    cusignm_cHalley(n, A, d_buffer, h_buffer, S);

    /*-----------------------------------------------------------------------------
     * check sign function ||S^2 - I||_F
     *-----------------------------------------------------------------------------*/
    float fronrmdiff = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cuComplex sum = cuComplex{0.0f, 0.0f};
            for (int k = 0; k < n; ++k) {
                sum = cuCaddf(sum, cuCmulf(S[i + k * n], S[k + j * n]));
            }
            if (i == j) {
                sum = cuCsubf(sum, cuComplex{1.0f, 0.0f});
            }
            float diff = cuCabsf(sum);
            fronrmdiff += diff * diff;
        }
    }
    float error = sqrtf(fronrmdiff / sqrtf((float)n));
    printf("rel. error ||S^2 - I ||_F / ||I||_F = %e\n", error);
    if (error < 1e-4) {
        printf("Sign Function Approximation successful\n");
    } else {
        printf("Sign Function Approximation failed\n");
        ret = 1;
    }

    /*-----------------------------------------------------------------------------
     * clear memory
     *-----------------------------------------------------------------------------*/
    cudaFree(A);
    cudaFree(S);
    cudaFree(d_buffer);
    free(h_buffer);
    return ret;
}
