#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 1024

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) 
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numEdges)
  {
    uint32_t src_node = edgeSrc[tid];//u
    uint32_t dst_node = edgeDst[tid];//v
    uint32_t src = rowPtr[src_node];//u_ptr->P[u]
    uint32_t dst = rowPtr[dst_node];//v_ptr->P[v]
    uint32_t src_end = rowPtr[src_node+1];//u_end->P[u+1]
    uint32_t dst_end = rowPtr[dst_node+1];//v_end->P[v+1]

    uint64_t num_triangle = 0;
    int middle;

    int low = 0; 
    int high = (dst_end-dst)-1;

    while (high >= low)
    {
      middle = (low+high)/2;
      if (edgeDst[dst+middle]==edgeDst[src])
      {
        num_triangle += 1;
        break;
      }
      else if(edgeDst[dst+middle] < edgeDst[src])
      {
        low = middle + 1;
      }
      else
      {
        high = middle - 1;
      }
    }
    src += 1;
    dst += middle;

    int w1 = edgeDst[src], w2 = edgeDst[dst];

    while (src < src_end && dst < dst_end)
    {  //u_ptr<u_end && v_ptr<v_end
      if (w1 < w2){
        w1 = edgeDst[++src];//w1<-Zd[++u_ptr]
      }
      else if (w1 > w2){
        w2 = edgeDst[++dst];
      }
      else {
        w1 = edgeDst[++src];
        w2 = edgeDst[++dst];
        num_triangle += 1;
      }
    }
    triangleCounts[tid] = num_triangle;
  }
}

__global__ void reduct(uint64_t *input, int len) {

  __shared__ uint64_t temp[BLOCK_SIZE];
  int id = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
  temp[threadIdx.x] = id < len ? input[id] : 0;
  temp[threadIdx.x] += (id + BLOCK_SIZE)<len ? input[id + BLOCK_SIZE] : 0;
  
  int space = BLOCK_SIZE>>1;
  __syncthreads();
  while (space)
  {
    if (threadIdx.x < space)
    {
      temp[threadIdx.x] += temp[threadIdx.x + space];
    }
    space = space >> 1;
    __syncthreads();
  }
  // atomicAdd((unsigned long long int*)(&(input[len])),(unsigned long long int)*temp) ;
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long int*)&(input[len]),(unsigned long long int)temp[0]);
}


uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // REQUIRED

    //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
    // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
    // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
    pangolin::Vector<uint64_t>tri_nums = pangolin::Vector<uint64_t>(view.nnz()+1) ;

    tri_nums[view.nnz()] = 0;
    //@@ launch the linear search kernel here
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid ( ceil(((float)view.nnz()) / BLOCK_SIZE ));
    
    kernel_tc <<<dimGrid, dimBlock>>> (tri_nums.data(),  //!< per-edge triangle counts
                                 view.row_ind(),         //!< node ids for edge srcs
                                 view.col_ind(),         //!< node ids for edge dsts
                                 view.row_ptr(),          //!< source node offsets in edgeDst
                                 view.nnz());                  //!< how many edges to count triangles for
 
    cudaDeviceSynchronize();

    dim3 dimGrid2 ( ceil(((float)view.nnz()) / (BLOCK_SIZE*2) ));
    reduct <<<dimGrid2, dimBlock>>>(tri_nums.data(), view.nnz());
    cudaDeviceSynchronize();
    uint64_t total = (tri_nums.data())[view.nnz()] ;

    return total;

  } else if (2 == mode) {

    // OPTIONAL. See README for more details

    uint64_t total = 0;
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count

    return total;
  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
}
