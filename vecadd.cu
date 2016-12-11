/*
 * Cuda vector addition code.
 */

#include <iostream>
#include <cstdlib>
#include <cassert>
using namespace std;

#include <cuda_runtime.h> // CUDA include

__global__ void vecadd(const float* A, const float* B, float* C, int length) {

  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < length)
    C[id] = A[id] + B[id];
}

int main(int argc, char* argv[]) {

  if (argc != 2) {
    cerr<<"Too many or too few arguments - Usage : vecadd <size of vectors>"<<endl;
  }

  int vec_size = atoi(argv[1]);
  assert(vec_size > 0 && "Invalid input size");

  int data_size = vec_size * sizeof(float);

  float* A = NULL;
  float* B = NULL;
  float* C = NULL;

  /* Allocate cpu memory */
  A = (float*) malloc(data_size);
  B = (float*) malloc(data_size);
  C = (float*) malloc(data_size);
  assert(A != NULL && B != NULL && C != NULL && "Cannot alloc memory");

  /* Fill inputs :
   * if vec_size is 10, then fill A and B as follows,
   * A : 0 1 2 3 4 5 6 7 8 9
   * B : 9 8 7 6 5 4 3 2 1 0
   */
  for (int vec_iter = 0; vec_iter < vec_size; vec_iter++) {
    A[vec_iter] = vec_iter;
    B[vec_iter] = vec_size - vec_iter - 1;
  }

  float* gpuA = NULL;
  float* gpuB = NULL;
  float* gpuC = NULL;

  cudaError_t err;

  /* Allocate GPU memory */
  err = cudaMalloc(&gpuA, data_size);
  assert(err == cudaSuccess && "cudaMalloc fail");
  err = cudaMalloc(&gpuB, data_size);
  assert(err == cudaSuccess && "cudaMalloc fail");
  err = cudaMalloc(&gpuC, data_size);
  assert(err == cudaSuccess && "cudaMalloc fail");

  /* Load inputs to gpu memory */
  err = cudaMemcpy(gpuA, A, data_size, cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "cudaMemcpy fail");
  err = cudaMemcpy(gpuB, B, data_size, cudaMemcpyHostToDevice);
  assert(err == cudaSuccess && "cudaMemcpy fail");

  /* Execute function on GPU */
  dim3 block_dim(512, 1, 1);
  dim3 grid_dim((vec_size / 512) + 1, 1, 1);

  vecadd<<<grid_dim, block_dim>>>(gpuA, gpuB, gpuC, vec_size);

  cudaDeviceSynchronize(); // Wait till gpu completes execution

  /* Copy results back to CPU */
  err = cudaMemcpy(C, gpuC, data_size, cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess && "cudaMemcpy fail");

  /* Check results */
  /* All elements of C should be vec_size - 1 */
  bool vec_add_pass = true;
  for (int vec_iter = 0; vec_iter < vec_size; vec_iter++) {
    if(C[vec_iter] != vec_size - 1) { vec_add_pass = false; break; }
  }
  if (vec_add_pass) { cout<<"Vector addition pass"<<endl; }
              else  { cout<<"Vector addition fail"<<endl; }

  /* Free cpu memory */
  free(A);
  free(B);
  free(C);

  /* Free cuda memory */
  cudaFree(gpuA);
  cudaFree(gpuB);
  cudaFree(gpuC);

  return 0;
}