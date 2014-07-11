#include <stdio.h>
#include <stdlib.h>

__global__ void add(int a, int b, int *c) {
  *c = a + b;
}

int main(int argc, char *argv[]) {
  int c;
  int *dev_c;
  cudaError_t error = cudaMalloc((void **)&dev_c, sizeof(int));
  if(error != cudaSuccess) {
    printf("Memory could not be allocated on device\n");
    exit(EXIT_FAILURE);
  }
  add<<<1,1>>>(2, 7, dev_c);

  error = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  if(error != cudaSuccess) {
    printf("Could not copy from device to host\n");
    exit(EXIT_FAILURE);
  }
  printf("%d = 2 + 7\n", c);
  cudaFree(dev_c);

  return 0;
}
