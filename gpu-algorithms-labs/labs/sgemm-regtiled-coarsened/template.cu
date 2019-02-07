#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE
  __shared__ float Bds[TILE_SZ_RATIO][TILE_SZ_B];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * TILE_SZ_A + tx;
  int Col = bx * TILE_SZ_B;
  float C_matrix[TILE_SZ_B]={0};
  float r;
  int shared_x = tx % TILE_SZ_B;
  int shared_y = tx / TILE_SZ_B;
  int num_iteration = ceil(float(k)/TILE_SZ_RATIO);
  for (int iter=0;iter<num_iteration;iter++){
    if (iter * TILE_SZ_RATIO+shared_y <k && Col+shared_x<n){
      Bds[shared_y][shared_x] = B(iter*TILE_SZ_RATIO+shared_y,Col+shared_x);
    }
    else{
      Bds[shared_y][shared_x]=0;
    }
  
     __syncthreads();
    for (int reg=0;reg<TILE_SZ_RATIO;reg++){
      r = A(Row,reg+iter*TILE_SZ_RATIO);
      for (int ii=0;ii<TILE_SZ_B;ii++){
        C_matrix[ii] += r * Bds[reg][ii];
      }
    }
    __syncthreads();
  }
  for (int jj=0;jj<TILE_SZ_B;jj++){
    if (Row<m && Col+jj<n){
      C(Row,Col+jj) = C_matrix[jj];
    }
  }

  
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    dim3 BlockDim (TILE_SZ_A,1,1);
    dim3 GridDim (ceil(float(n)/TILE_SZ_B),ceil(float(m)/TILE_SZ_A),1);
    mysgemm<<<GridDim,BlockDim>>>(m, n, k, A, B,C); 


}
