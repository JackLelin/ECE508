#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"


__global__ void child_tc(uint64_t* num_triangle,
                        const uint32_t *const edgeDst,
                        const uint32_t srcStart,
                        const uint32_t dstStart,
                        uint32_t num_Src,
                        uint32_t num_Dst)


{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_Src)
  {
    for (int j = 0; j < num_Dst; j ++)
    {
        if (edgeDst[idx + srcStart] == edgeDst[dstStart + j])
            atomicAdd((unsigned long long int*)num_triangle,(unsigned long long int)1);
    }
    
  }
}


#define BLOCK_SIZE 1024

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numEdges)
    {
        int src_start = rowPtr[edgeSrc[idx]]; // row start ptr
        int src_end = rowPtr[edgeSrc[idx] + 1];
        int dst_start = rowPtr[edgeDst[idx]]; // col start ptr
        int dst_end = rowPtr[edgeDst[idx]+1];

        int NumSrcEdge = src_end - src_start;
        int NumDstEdge = dst_end - dst_start;

        triangleCounts[idx] = 0; 
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        child_tc <<<256,(NumSrcEdge-1)/256+1,0,s>>> (&triangleCounts[idx],edgeDst,src_start,dst_start,NumSrcEdge,NumDstEdge);

        cudaDeviceSynchronize();
        __syncthreads();
    }

  // Determine the source and destination node for the edge
  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  // Determine how many elements of those two arrays are common
}

__global__ void recursive_k(int depth) {
    int * a;
    cudaMalloc((void**)&a, sizeof(int));
    *a = 1;
  // up to depth 6
  if(depth == 6)
    return;
  // launch 1 kernel (2 if depth == 3)
  if(threadIdx.x == 0) {
    recursive_k <<< 1, 1 >>> (depth + 1);
    if(depth == 3) {
      cudaDeviceSynchronize();
      recursive_k <<< 1, 1 >>> (depth + 1);
    }
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
    
    // launch from host
    // recursive_k <<< 1, 1 >>> (0);
    // printf("============================================================================recursive_k\n");

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