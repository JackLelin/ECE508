#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 32
__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {

  // INSERT KERNEL CODE HERE
#define in(ii,jj,kk) A0[((kk)*ny+jj)*nx+ii]
#define out(ii,jj,kk) Anext[((kk)*ny+jj)*nx+ii]
__shared__ int ds_A[TILE_SIZE][TILE_SIZE];
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int i = bx * (TILE_SIZE) + tx, j = by * (TILE_SIZE) + ty;
int bottom, current, top;

if (i<nx && j<ny){
  bottom = in(i,j,0);
  current = in(i,j,1);
  top = in(i,j,2);
  
  }

ds_A[ty][tx] = current;
__syncthreads();
for (int k=1;k<nz-1;k++){
  if (i>0 && i<nx-1 && j>0 && j<ny-1){
    out(i,j,k) = bottom + top + ((tx>0)? ds_A[ty][tx-1]:in(i-1,j,k))+
        ((tx<blockDim.x-1)? ds_A[ty][tx+1]:in(i+1,j,k))+
        ((ty>0)? ds_A[ty-1][tx]:in(i,j-1,k))+
        ((ty<blockDim.y-1)? ds_A[ty+1][tx]:in(i,j+1,k))-6*current;
    }
    __syncthreads();
    
    bottom = current; ds_A[ty][tx] = top; current = top; 
    // __syncthreads();
    if (i<nx && j<ny && k<nz-2){
      top = in(i,j,k+2);
       
      }
    __syncthreads();
    
  }
  
 

#undef A0
#undef Anext  

}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  // INSERT CODE HERE
int BLOCKSIZE = 32;
dim3 dimBlock (BLOCKSIZE,BLOCKSIZE,1);
dim3 dimGrid (ceil(float(nx)/BLOCKSIZE),ceil(float(ny)/BLOCKSIZE),1);

kernel<<<dimGrid,dimBlock>>>(A0,Anext,nx,ny,nz);
}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
