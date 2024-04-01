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

#pragma once

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

int cusignm_sNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_dNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_cNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_zNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);

int cusignm_sNewton(const int n, const float *A, void *d_buffer, void *h_buffer, float *S);
int cusignm_dNewton(const int n, const double *A, void *d_buffer, void *h_buffer, double *S);
int cusignm_cNewton(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *S);
int cusignm_zNewton(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *S);

int cusignm_sHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_dHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_cHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cusignm_zHalleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);

int cusignm_sHalley(const int n, const float *A, void *d_buffer, void *h_buffer, float *S);
int cusignm_dHalley(const int n, const double *A, void *d_buffer, void *h_buffer, double *S);
int cusignm_cHalley(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *S);
int cusignm_zHalley(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *S);

#ifdef __cplusplus
}
#endif
