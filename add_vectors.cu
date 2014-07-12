#include <stdio.h>
#include <stdlib.h>

#define N 10000

void handleError(cudaError_t error) {
  if(error != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if(tid < N) {
      c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char *argv[]) {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  handleError(cudaMalloc((void **) &dev_a, sizeof(int) * N));
  handleError(cudaMalloc((void **) &dev_b, sizeof(int) * N));
  handleError(cudaMalloc((void **) &dev_c, sizeof(int) * N));

  for(int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  handleError(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
  handleError(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  handleError(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

  for(int i = 0; i <  N; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
