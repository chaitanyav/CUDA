/*
 * Author: NagaChaitanya Vellanki
 *
 ************************************************************************************************
 ************************************************************************************************ 
 * sample output: 
 * 1 devices on this machine
* printing device information
*
* Device Name: GeForce GTX 550 Ti
* Total Global Memory: 1073741824
* Shared Memory per Block: 49152
* Registers per Block: 32768
* Warp Size: 32
* Mem Pitch: 2147483647
* Max Threads per Block: 1024
* Max Threads Dimension: 1024 1024 1024
* Max Grid Size: 65535 65535 65535
* Total Constant Memory: 65536
* Major version: 2, Minor Version: 1
* Texture Alignment: 512
* Multi Processor count: 4
* Runtime Limit of Kernels Executed: 1
* Integrated or Discrete GPU (0 means discrete, 1 means integrated): 0
* Can map host memory into device address space: 1
* Compute Mode(default, exclusive or prohibited): 0
* Max 1D Texture size: 65536
* Max Texture 2D Dimensions: 65536 65535
* Max Texture 3D Dimensions: 2048 2048 2048
* Concurrent Kernels: 1
* Press any key to continue . . .
**************************************************************************************************
**************************************************************************************************
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void handleError(cudaError_t err, const char *file, int line) {
 if(err != cudaSuccess) {
 printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
				file, line);
		exit( EXIT_FAILURE );
 }
}

int main(int argc, char *argv[]) {
int count;
int i;
cudaDeviceProp prop;

handleError(cudaGetDeviceCount(&count), __FILE__, __LINE__);

printf("%d devices on this machine\n", count);

for(i = 0; i < count; i++) {
	handleError(cudaGetDeviceProperties(&prop, i), __FILE__, __LINE__);
	printf("printing device information\n\n");
	printf("Device Name: %s\n", prop.name);
	printf("Total Global Memory: %d\n", prop.totalGlobalMem);
	printf("Shared Memory per Block: %d\n", prop.sharedMemPerBlock);
	printf("Registers per Block: %d\n", prop.regsPerBlock);
	printf("Warp Size: %d\n", prop.warpSize);
	printf("Mem Pitch: %d\n", prop.memPitch);
	printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
	printf("Max Threads Dimension: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[1]);
	printf("Max Grid Size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[1]);
	printf("Total Constant Memory: %d\n", prop.totalConstMem);
	printf("Major version: %d, Minor Version: %d\n", prop.major, prop.minor);
	printf("Texture Alignment: %d\n", prop.textureAlignment);
	printf("Multi Processor count: %d\n", prop.multiProcessorCount);
	printf("Runtime Limit of Kernels Executed: %d\n", prop.kernelExecTimeoutEnabled);
	printf("Integrated or Discrete GPU (0 means discrete, 1 means integrated): %d\n", prop.integrated);
	printf("Can map host memory into device address space: %d\n", prop.canMapHostMemory);
	printf("Compute Mode(default, exclusive or prohibited): %d\n", prop.computeMode);
	printf("Max 1D Texture size: %d\n", prop.maxTexture1D);
	printf("Max Texture 2D Dimensions: %d %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
	printf("Max Texture 3D Dimensions: %d %d %d\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
	printf("Concurrent Kernels: %d\n", prop.concurrentKernels);
}

 exit(EXIT_SUCCESS);
}