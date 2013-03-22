/*
* Author: NagaChaitanya Vellanki
*
*
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
cudaDeviceProp prop;
int device;

handleError(cudaGetDevice(&device), __FILE__, __LINE__);
printf("Device is %d\n", device);

memset(&prop, 0, sizeof(cudaDeviceProp));
prop.major = 2;
prop.minor = 1;
handleError(cudaChooseDevice(&device, &prop), __FILE__, __LINE__);
printf("Chosen Device is %d\n", device);

handleError(cudaSetDevice(device), __FILE__, __LINE__);

exit(EXIT_SUCCESS);
}