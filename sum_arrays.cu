#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void handleError(cudaError_t error, int lineno) {
  if(error != cudaSuccess) {
    printf("Error: %s %d\n", __FILE__, lineno);
    printf("code: %d, reason %s\n", error, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void initializeData(float *ptr, int size) {
 time_t t;
 srand((unsigned) time(&t));
 for(int pos = 0; pos < size; pos++) {
   ptr[pos] = (float) (rand() & 0xFF) / 10.0F;
 }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
  int id = threadIdx.x;
  C[id] = A[id] + B[id];
}

int main(int argc, char *argv[]) {
  int dev = 0;
  cudaSetDevice(dev);

  int nElem = 1024;
  size_t nBytes = nElem * sizeof(float);
  float *h_A, *h_B, *h_C;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);
  initializeData(h_A, nElem);
  initializeData(h_B, nElem);
  float *d_A, *d_B, *d_C;
  handleError(cudaMalloc((float **)&d_A, nBytes), __LINE__);
  handleError(cudaMalloc((float **)&d_B, nBytes), __LINE__);
  handleError(cudaMalloc((float **)&d_C, nBytes), __LINE__);

  handleError(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice), __LINE__);
  handleError(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice), __LINE__);

  dim3 block(nElem);
  dim3 grid(nElem/block.x);

  sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);
  handleError(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost),
      __LINE__);

  for(int pos = 0; pos < nElem; pos++) {
    printf("%f + %f = %f\n", h_A[pos], h_B[pos], h_C[pos]);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  return 0;
}
