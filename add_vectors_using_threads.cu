#include <stdio.h>
#include <stdlib.h>


#define N 500

__global__ void add(int *a, int *b, int *c) {
  int tid = threadIdx.x;
  if(tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

void handleError(cudaError_t error, int lineNo) {
  if(error != cudaSuccess) {
    printf("Error: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, lineNo);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  handleError(cudaMalloc((void **)&dev_a, N * sizeof(int)), __LINE__);
  handleError(cudaMalloc((void **)&dev_b, N * sizeof(int)), __LINE__);
  handleError(cudaMalloc((void **)&dev_c, N * sizeof(int)), __LINE__);

  for(int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  handleError(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
  handleError(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
  handleError(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);

  add<<<1,N>>>(dev_a, dev_b, dev_c);
  handleError(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);
  for(int i = 0; i < N; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
