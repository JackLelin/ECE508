#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)
#define CEILDIV(X,Y) (X-1)/Y+1

__host__ __device__ static inline int A_bs( int row, int col, 
                                            const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                            const uint32_t *const rowPtr          //!< source node offsets in edgeDst
) 
{
  int low = rowPtr[row]; 
  int high = rowPtr[row+1]-1;
  int middle;
  while (high >= low)
  {
    middle = (low+high)/2;
    if (edgeDst[middle]==col)
      return 1;
    else if(edgeDst[middle] < col)
      low = middle + 1;
    else
      high = middle - 1;
  }
  return 0;
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const uint64_t numRows
)
{

    __shared__ int N_cache [TILE_SZ_B][TILE_SZ_RATIO]; // column major

    int c_x = threadIdx.y%TILE_SZ_B;
    int c_y = threadIdx.y/TILE_SZ_B;

    int idx = TILE_SZ_B * blockIdx.x + c_x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int M_element = 0;
    int P_elements[TILE_SZ_B] = {0};

    int idk_A;
    int idk_B;

    for (int i_chunk = 0; i_chunk < CEILDIV(numRows,TILE_SZ_RATIO); i_chunk++)
    {
        idk_B = i_chunk * TILE_SZ_RATIO + c_y;
        N_cache[c_x][c_y] = (idx < numRows && idk_B < numRows)? A_bs(idk_B, idx,edgeDst,rowPtr):0; // Load N matrix
        __syncthreads();
        for (int i_step = 0; i_step<TILE_SZ_RATIO; i_step++)
        {
            idk_A = i_chunk * TILE_SZ_RATIO + i_step;
            M_element = (idy < numRows && idk_A<numRows)?A_bs(idy,idk_A,edgeDst,rowPtr):0; // Load M matrix
            for (int ii = 0; ii < TILE_SZ_B; ii ++) {P_elements[ii] += M_element * N_cache[ii][i_step];}
        }
    }
    __syncthreads();
    for (int ii = 0; ii < TILE_SZ_B; ii ++)
    {
      if(idy < numRows && (TILE_SZ_B * blockIdx.x+ii) < numRows) 
      {
        if( A_bs(idy,TILE_SZ_B * blockIdx.x+ii,edgeDst,rowPtr))
          {          
            atomicAdd((unsigned long long int*)triangleCounts,(unsigned long long int)P_elements[ii]);
          }      
      }
    }
}


uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // REQUIRED

    //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
    // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
    // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
    //@@ launch the linear search kernel here
    pangolin::Vector <uint64_t> edge_array (1,0);
    
    edge_array.data()[0] = 0;

    int blockx = CEILDIV(view.num_rows(),TILE_SZ_B);
    int blocky = CEILDIV(view.num_rows(),TILE_SZ_A); 

    dim3 DimGrid(blockx,blocky,1);
    dim3 DimBlock(1,TILE_SZ_A,1);

    //pangolin::Vector<uint64_t>edge_array = pangolin::Vector<uint64_t>(view.nnz()) ;

    kernel_tc <<<DimGrid,DimBlock>>>(edge_array.data(), view.row_ind(),view.col_ind(),view.row_ptr(),view.num_rows());

    cudaDeviceSynchronize();
    return edge_array.data()[0];

  } else if (2 == mode) {

    // OPTIONAL. See README for more details
    pangolin::Vector<uint64_t> edge_array(view.nnz(),0);
    dim3 dimBlock(512);
    dim3 dimGrid(ceil(float(view.nnz())/512),1,1);
    kernel_tc<<<dimGrid,dimBlock>>>(edge_array.data(),view.row_ind(),view.col_ind(),view.row_ptr(),view.nnz());
    cudaDeviceSynchronize();
    uint64_t total = 0;
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
    for (int i=0;i<view.nnz();i++){
      total += edge_array[i];
    }
    return total;
  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
}
