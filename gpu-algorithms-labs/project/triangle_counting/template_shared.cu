#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define CEILDIV(X,Y) (X-1)/Y+1

#define SHARED_SIZE 2048 
#define BLOCK_SIZE 1024
#define GRID_SIZE 128
#define THRESH 512

__global__ static void kernel_shared_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint64_t numRows,         //!< node ids for edge srcs
                                 const uint32_t *const edgeSrc,
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr         //!< source node offsets in edgeDst
) 
{
  __shared__ int edgeDst_S[SHARED_SIZE]; 
  for (int rid = blockIdx.x; rid < numRows; rid += GRID_SIZE)
  {

  }
  int idx = threadIdx.x;
  while ( idx < numSrc && idx < SHARED_SIZE)
  {
    edgeDst_S[idx] = edgeDst[srcStart + idx];
    idx += blockDim.x;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < numSrc; idx += blockDim.x)
  {
    int eid = srcStart + idx;

    uint32_t dst_node = edgeDst[eid];//v

    uint32_t src = 0;//u_ptr->P[u]
    uint32_t dst = rowPtr[dst_node];//v_ptr->P[v]
    uint32_t dst_end = rowPtr[dst_node+1];//v_end->P[v+1]
    
    uint64_t num_triangle = 0;

    int w1 = edgeDst_S[src], w2 = edgeDst[dst];//w1->Zd[u_ptr] w2->Zd[v_ptr]
    while (src < numSrc && dst < dst_end)
    {  
      if (w1 < w2)
      {
        src++;
        if (src < SHARED_SIZE)
        {            
          w1 = edgeDst_S[src];//w1<-Zd[++u_ptr]
        }          
        else
        {
          w1 = edgeDst[src + srcStart];
        }
      }
      else if (w1 > w2)
      {
        dst++;
        w2 = edgeDst[dst]; 
      }
      else 
      {
        src++;
        dst++;
        if (src < SHARED_SIZE)
        {            
          w1 = edgeDst_S[src];//w1<-Zd[++u_ptr]
        }          
        else
        {
          w1 = edgeDst[src + srcStart];
        }
        w2 = edgeDst[dst];
        num_triangle += 1;
      }
    }
    triangleCounts[eid] = num_triangle;
  }
}


uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) 
{
  if (mode == 1) {

    uint64_t total = 0; 

    uint64_t * tri_nums;
    cudaMalloc((void**)&tri_nums,(view.nnz())*sizeof(uint64_t));
    cudaMemset(tri_nums,0,view.nnz()*sizeof(uint64_t));
    
    cudaDeviceSynchronize();
    int srcStart,numSrc;

    const uint32_t * edgeDst = view.col_ind();
    const uint32_t * rowPtr = view.row_ptr();  


    kernel_shared_tc <<<GRID_SIZE, BLOCK_SIZE>>> 
                                    (tri_nums,  //!< per-edge triangle counts
                                     view.num_rows(),
                                     view.row_ind(),
                                     view.col_ind(),         //!< node ids for edge dsts
                                     view.row_ptr());                  //!< how many edges to count triangles for 

    for (int rid = 0; rid <view.num_rows(); rid++)
    {
      srcStart = view.row_ptr()[rid];
      numSrc = view.row_ptr()[rid+1] - srcStart;


      if (numSrc < THRESH )
      {
        for(int idx = 0; idx < numSrc; idx++)
        {
          int eid = idx + srcStart;
          // uint32_t src_node = view.row_ind()[eid];
          uint32_t dst_node = edgeDst[eid];//v

          uint32_t src = srcStart;//u_ptr->P[u]
          uint32_t src_end = srcStart + numSrc;//u_ptr->P[u]
          uint32_t dst = rowPtr[dst_node];//v_ptr->P[v]
          uint32_t dst_end = rowPtr[dst_node+1];//v_end->P[v+1]
          
          int w1 = edgeDst[src], w2 = edgeDst[dst];//w1->Zd[u_ptr] w2->Zd[v_ptr]
          while (src < src_end && dst < dst_end)
          {  //u_ptr<u_end && v_ptr<v_end
            if (w1 < w2)
            {
              w1 = edgeDst[++src];//w1<-Zd[++u_ptr]
            }
            else if (w1 > w2)
            {
              w2 = edgeDst[++dst];
            }
            else 
            {
              w1 = edgeDst[++src];
              w2 = edgeDst[++dst];
              total += 1;
            }
          }
        }
      }
    }

    

    uint64_t * tri_nums_H = new uint64_t[view.nnz()];
    cudaMemcpy(tri_nums_H,tri_nums,view.nnz()*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    for (int ii = 0; ii < view.nnz(); ii++)
      total += tri_nums_H[ii];

    delete(tri_nums_H);
    cudaFree(tri_nums_H);
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
