#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int numDevices;
  cudaDeviceProp prop;
  cudaError_t errorNum = cudaGetDeviceCount(&numDevices);
  if(errorNum != cudaSuccess) {
    printf("Could not get device count\n");
    exit(EXIT_FAILURE);
  }

  printf("Number of CUDA capable devices on this machine is %d\n", numDevices);
  for(int device = 0; device < numDevices; device++) {
    errorNum = cudaGetDeviceProperties(&prop, device);
    if(errorNum == cudaSuccess) {
      printf("Device name: %s\n", prop.name);
      printf("Total Global Memory(Bytes): %lu\n", prop.totalGlobalMem);
      printf("Shared Memory per Block(Bytes): %lu\n", prop.sharedMemPerBlock);
      printf("Register per Block: %d\n", prop.regsPerBlock);
      printf("Number of Thread in a Warp: %d\n", prop.warpSize);
      printf("Maximum pitch allowed for memory copies: %lu\n", prop.memPitch);
      printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
      printf("Max threads across each dim in Block: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf("Max blocks across each dim in grid: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
      printf("Available constant memory: %lu\n", prop.totalConstMem);
      printf("Major version of compute capability: %d\n", prop.major);
      printf("Minor version of compute capability: %d\n", prop.minor);
      printf("Device texture alignment requirement: %lu\n", prop.textureAlignment);
      printf("Is Device overlap: %d\n", prop.deviceOverlap);
      printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
      printf("Is runtime limit on kernels: %d\n", prop.kernelExecTimeoutEnabled);
      printf("Is integrated: %d\n", prop.integrated);
      printf("Can map host memory: %d\n", prop.canMapHostMemory);
      printf("Compute Mode: %d\n", prop.computeMode);
      printf("Max 1d textures: %d\n", prop.maxTexture1D);
      printf("Max 2d texture: %d %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
      printf("Max 3d texture: %d %d %d\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
      printf("Can support concurrent kernels: %d\n", prop.concurrentKernels);
    }
  }
  return 0;
}
