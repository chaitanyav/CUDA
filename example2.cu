#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c) {
*c = a + b;
}

void handleError(cudaError_t err, const char *file, int line) {
 if(err != cudaSuccess) {
 printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
 }
}

int main(int argc, char *argv[]) {
 int c;
 int *dev_c;

 handleError(cudaMalloc((void **) &dev_c, sizeof(int)), __FILE__, __LINE__);

 add<<<1,1>>>(2,7, dev_c);

 handleError(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

 printf("2 + 7 = %d\n", c);
 cudaFree(dev_c);

 exit(EXIT_SUCCESS);
}