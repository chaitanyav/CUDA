#include <stdio.h>
#include <stdlib.h>

void handleError(cudaError_t error) {
  if(error != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  cudaDeviceProp prop;
  int dev;
  handleError(cudaGetDevice(&dev));
  printf("Current Device ID is %d\n", dev);

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 3;
  prop.minor = 0;

  handleError(cudaChooseDevice(&dev, &prop));
  printf("The closest device to revision 3.0 is %d\n", dev);
  handleError(cudaSetDevice(dev));
  return 0;
}
