#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"


__global__ void child_tc(uint64_t* num_triangle,
                        const uint32_t *const edgeDst,
                        const uint32_t *const rowPtr,
                        const uint32_t edgeStart,
                        uint32_t num_edge)


{
  extern __shared__ int edgeDst_S[];
  for (int i = blockIdx.x; i < num_edge; i += blockDim.x)
  {
    edgeDst_S[i] = edgeDst[edgeStart + i];
  }
  __syncthreads();
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int eid = edgeStart + idx;

  uint32_t dst_node = edgeDst_S[eid];//v


  uint32_t dst = rowPtr[dst_node];//v_ptr->P[v]
  uint32_t dst_end = rowPtr[dst_node+1];//v_end->P[v+1]
  
  uint64_t num_triangle = 0;
  uint32_t src = 0;//u_ptr->P[u]

  int w1 = edgeDst_S[src], w2 = edgeDst[dst];//w1->Zd[u_ptr] w2->Zd[v_ptr]
  while (src < num_edge && dst < dst_end)
  {  //u_ptr<u_end && v_ptr<v_end
    if (w1 < w2)
    {
      w1 = edgeDst_S[++src];//w1<-Zd[++u_ptr]
    }
    else if (w1 > w2)
    {
      w2 = edgeDst[++dst];
    }
    else 
    {
      w1 = edgeDst_S[++src];
      w2 = edgeDst[++dst];
      num_triangle += 1;
    }
  }
  triangleCounts[eid] = num_triangle;
}


#define BLOCK_SIZE 1024

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) 
{
    int idx = blockIdx.x;
    int edgeStart = rowPtr[idx]
    int num_edge = (idx == gridDim.x)?numEdges:rowPtr[idx + 1] - edgeStart;

    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    child_tc <<<1024,(num_dst-1)/(1024)+1,num_dst,s>>> (triangleCounts,edgeDst,rowPtr,edgeStart,num_edge);


  // Determine the source and destination node for the edge
  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  // Determine how many elements of those two arrays are common
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


uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) 
{
  if (mode == 1) {

    // REQUIRED

    //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
    // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
    // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
    pangolin::Vector<uint64_t>tri_nums = pangolin::Vector<uint64_t>(view.nnz()+1) ;

    tri_nums[view.nnz()] = 0;
    //@@ launch the linear search kernel here
    dim3 dimBlock(BLOCK_SIZE);
    // dim3 dimGrid ( ceil(((float)view.nnz()) / BLOCK_SIZE ));

    kernel_tc <<<1, view.num_rows()>>> (tri_nums.data(),  //!< per-edge triangle counts
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