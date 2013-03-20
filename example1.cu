#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(void) {}

int main(int argc, char *argv[]) {
kernel<<<1,1>>>();
printf("Hello World\n");
return 0;
}