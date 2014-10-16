#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void handleError(cudaError_t error, int lineno) {
  if (error != cudaSuccess) {
    printf("Error: %s:%d\n", __FILE__, lineno);
    printf("Code: %d, Reason: %s\n", error, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void initializeInt(int *iptr, int size) {
  for(int i = 0; i < size; i++) {
    iptr[i] = i;
  }
}

void printMatrix(int *iptr, const int nx, const int ny) {
  int *C = iptr;
    for(int i = 0; i < nx; i++) {
      for(int j = 0; j < ny; j++) {
        printf("%3d\n", C[j]);
      }
        C += ny;
        printf("\n");
    }
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
  int ix = threadIdx.x  + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;

  printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  handleError(cudaGetDeviceProperties(&deviceProp, dev), __LINE__);
  printf("Using device %d:%s\n", dev, deviceProp.name);
  handleError(cudaSetDevice(dev), __LINE__);

  int nx = 8;
  int ny = 6;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  int *h_A;
  h_A = (int *)malloc(nBytes);
  initializeInt(h_A, nBytes);
  printMatrix(h_A, nx, ny);

  int *d_A;
  handleError(cudaMalloc((void **)&d_A, nBytes), __LINE__);
  handleError(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice), __LINE__);

  dim3 block(4, 2);
  dim3 grid((nx + block.x - 1)/ block.x, (ny + block.y - 1)/ block.y);
  printThreadIndex<<<grid, block>>>(d_A, nx, ny);
  handleError(cudaDeviceSynchronize(), __LINE__);
  cudaFree(d_A);
  free(h_A);
  cudaDeviceReset();
  return 0;
}
