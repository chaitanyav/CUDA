#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void handleError(cudaError_t error, int lineno) {
    if(error != cudaSuccess) {
      printf("Error %s:%d\n", __FILE__, lineno);
      printf("Code: %d, Reason: %s\n", error, cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }
}

void initializeData(float *iptr, int size) {
    time_t t;
    srand((unsigned int) time(&t));

    for(int i = 0; i < size; i++) {
      iptr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void sumMatrix(float *A, float *B, float *C, int nx, int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = ix + iy * nx;
  if(ix < nx && iy < ny) {
    C[idx] = A[idx] + B[idx];
  }
}
int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  handleError(cudaGetDeviceProperties(&deviceProp, dev), __LINE__);
  handleError(cudaSetDevice(dev), __LINE__);

  int nx = 1 << 11;
  int ny = 1 << 11;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  float *h_A, *h_B, *gpuRef;
  h_A = (float *) malloc(nBytes);
  h_B = (float *) malloc(nBytes);
  gpuRef = (float *) malloc(nBytes);
  memset(gpuRef, 0, nBytes);

  initializeData(h_A, nxy);
  initializeData(h_B, nxy);

  float *d_A, *d_B, *d_C;
  handleError(cudaMalloc((void **)&d_A, nBytes), __LINE__);
  handleError(cudaMalloc((void **)&d_B, nBytes), __LINE__);
  handleError(cudaMalloc((void **)&d_C, nBytes), __LINE__);

  handleError(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice), __LINE__);
  handleError(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice), __LINE__);

  int dimx = 32;
  int dimy = 32;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1)/ block.x, (ny + block.y - 1)/ block.y);
  sumMatrix<<<grid, block>>>(d_A, d_B, d_C, nx, ny);

  handleError(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost), __LINE__);
  for(int i = 0; i < nx; i++) {
    for(int j = 0; j < ny; j++) {
      printf("%.2f + %.2f = %.2f\n", h_A[i * nx + j], h_B[i * nx + j], gpuRef[i * nx + j]);
    }
    printf("\n");
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(gpuRef);
  cudaDeviceReset();

  return 0;
}
