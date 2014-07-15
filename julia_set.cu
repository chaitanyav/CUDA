//------------------------------------------------------------------------------------------------------------------------------------
// To compile use this following command
// nvcc julia_set.cu -lglfw -Xlinker -framework,Cocoa,-framework,OpenGL,-framework,IOKit,-framework,CoreVideo
//
// Reference: Title: CUDA by Example: An Introduction to General-Purpose GPU Programming 1st Author: Jason Sanders, Edward Kandrot Pages: 312 Publisher: Addison-Wesley Professional ©2010 ISBN: 0131387685 9780131387683
//-----------------------------------------------------------------------------------------------------------------------------------
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 2000

struct cuComplex {
  float r;
  float i;

  __device__ cuComplex(float a, float b) : r(a), i(b) {
  }

  __device__ float magnitude2(void) {
    return r * r + i * i;
  }

  __device__ cuComplex operator*(const cuComplex& a) {
    return cuComplex(r * a.r - i * a.i, i * a.r  + r * a.i);
  }

  __device__ cuComplex operator+(const cuComplex& a) {
    return cuComplex(r + a.r, i + a.i);
  }
};

__device__ int julia(int x, int y) {
  const float scale = 1.5;
  int dim_2 = DIM / 2;
  float jx = scale * (float) (dim_2  - x) / dim_2;
  float jy = scale * (float) (dim_2  - y) / dim_2;

  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);

  for(int i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000) {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(unsigned char *ptr) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;
  int juliavalue = julia(x, y);
  ptr[offset * 4 + 0] = 127 * juliavalue;
  ptr[offset * 4 + 1] = 255 * juliavalue;
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 255;
}

struct CPUBitmap {
  unsigned char    *pixels;
  int     x, y;
  void    *dataBlock;

  CPUBitmap( int width, int height, void *d = NULL  ) {
    pixels = new unsigned char[width * height * 4];
    x = width;
    y = height;
    dataBlock = d;
  }

  ~CPUBitmap() {
    delete [] pixels;
  }

  unsigned char* get_ptr( void  ) const   { return pixels;  }
  long image_size( void  ) const { return x * y * 4;  }

  void display_and_exit() {
    CPUBitmap**   bitmap = get_bitmap_ptr();
    *bitmap = this;
    GLFWwindow* window;

    if(!glfwInit()) {
      printf("Failed on GLFW init\n");
      exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(x, y, "Bitmap", NULL, NULL);
    if(!window) {
      glfwTerminate();
      printf("Failed to create window\n");
      exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    while(!glfwWindowShouldClose(window)) {
      CPUBitmap*   bitmap =  *(get_bitmap_ptr());
      glClearColor( 0.0, 0.0, 0.0, 1.0  );
      glClear( GL_COLOR_BUFFER_BIT );
      glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
      glFlush();
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
    glfwTerminate();
  }

  // static method used for glut callbacks
  static CPUBitmap** get_bitmap_ptr( void  ) {
    static CPUBitmap   *gBitmap;
    return &gBitmap;
  }

  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
  {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
      glfwSetWindowShouldClose(window, GL_TRUE);
      glfwTerminate();
      exit(EXIT_SUCCESS);
    }
  }
};

void handleError(cudaError_t error, int lineNo) {
  if(error != cudaSuccess) {
    printf("Error: %s\n in file %s at line no %d\n",
    cudaGetErrorString(error), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *dev_bitmap;
  handleError(cudaMalloc((void **) &dev_bitmap, bitmap.image_size()), __LINE__);
  dim3 grid(DIM, DIM);
  kernel<<<grid,1>>>(dev_bitmap);
  handleError(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,bitmap.image_size(), cudaMemcpyDeviceToHost), __LINE__);
  bitmap.display_and_exit();
  handleError(cudaFree(dev_bitmap), __LINE__);
  return 0;
}
