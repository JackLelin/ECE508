#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <float.h>
#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}
__device__ void merge_sequential_circular(float* A, int m, float* B, int n, float *C, int A_S_start,int B_S_start){
int i=0,j=0,k=0;
  while((i<m)&&(j<n)){
    int i_cir = (A_S_start+i>=TILE_SIZE)?A_S_start+i-TILE_SIZE:A_S_start+i;
    int j_cir = (B_S_start+j>=TILE_SIZE)?B_S_start+j-TILE_SIZE:B_S_start+j;
    if (A[i_cir]<=B[j_cir]){
       C[k++] = A[i_cir];
       i++;
    }
    else{
      C[k++] = B[j_cir];
      j++;
    }
  }
  if (i==m){
    for(;j<n;j++){
      int j_cir=(B_S_start+j>=TILE_SIZE)?B_S_start+j-TILE_SIZE:B_S_start+j;
      C[k++] = B[j_cir];
    }
  }
 else{
  for (;i<m;i++){
    int i_cir=(A_S_start+i>=TILE_SIZE)?A_S_start+i-TILE_SIZE:A_S_start+i;
    C[k++] = A[i_cir];
  }
 }
}
__device__ int co_rank(int k, float *A, int m, float *B, int n){
  int i = k<m?k:m;
  int j = k-i;
  int i_low = 0>(k-n)?0:k-n;
  int j_low = 0>(k-m)?0:k-m;
  int delta;
  bool active = true;
  while (active){
    if (i>0&&j<n&&A[i-1]>B[j]){
      delta = ((i-i_low+1)>>1);//delta = ceil_div(i-i_low,2);
      j_low = j;
      j = j + delta;
      i = i - delta;
    }
    else if (j>0 && i<m && B[j-1]>=A[i]){
      delta = ((j-j_low+1)>>1);// delta = ceil_div(j-j_low,2);
      i_low = i;
      i = i + delta;
      j = j - delta;
    }
    else{
      active = false;
    }
  }
  return i;
}
__device__ int co_rank_circular(int k,float *A,int m,float *B,int n,int A_S_start,int B_S_start){
    int i = k<m?k:m;
    int j =k-i;
    int i_low = 0>(k-n)?0:k-n;
    int j_low = 0>(k-m)?0:k-m;
    int delta;
    bool active = true;
    while (active){
      int i_cir = (A_S_start+i>=TILE_SIZE)?A_S_start+i-TILE_SIZE:A_S_start+i;
      int i_m_1_cir = (A_S_start+i-1>=TILE_SIZE)?A_S_start+i-1-TILE_SIZE:A_S_start+i-1;
      int j_cir = (B_S_start+j>=TILE_SIZE)?B_S_start+j-TILE_SIZE:B_S_start+j;
      int j_m_1_cir = (B_S_start+j-1>=TILE_SIZE)?B_S_start+j-1-TILE_SIZE:B_S_start+j-1;
      if (i>0&&j<n&&A[i_m_1_cir]>B[j_cir]){
        delta = ((i-i_low+1)>>1);
        j_low=j;
        i=i-delta;
        j=j+delta;
      }
      else if(j>0&&i<m&&B[j_m_1_cir]>=A[i_cir]){
        delta = ((j-j_low+1)>>1);
        i_low = i;
        i=i+delta;
        j=j-delta;
      }
      else{
        active=false;
      }
    }
    return i;

}
/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int k_curr = tid * ceil_div(A_len+B_len,blockDim.x * gridDim.x);
    int k_next = min((tid+1)*ceil_div(A_len+B_len,blockDim.x * gridDim.x),A_len+B_len);

    int i_curr = co_rank(k_curr,A,A_len,B,B_len);
    int i_next = co_rank(k_next,A,A_len,B,B_len);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr],i_next-i_curr,&B[j_curr],j_next-j_curr,&C[k_curr]);

}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */

__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    __shared__ float shareAB[2*TILE_SIZE];
    float *A_S=&shareAB[0];
    float *B_S=&shareAB[TILE_SIZE];
    int C_curr = blockIdx.x * ceil_div(A_len+B_len,gridDim.x);
    int C_next = min((blockIdx.x+1)*ceil_div(A_len+B_len,gridDim.x),A_len+B_len);
    __shared__ int A_share[2];
    if (threadIdx.x==0){
      A_share[0] = co_rank(C_curr,A,A_len,B,B_len);
      A_share[1] = co_rank(C_next,A,A_len,B,B_len);
    }
    __syncthreads();

    int A_curr = A_share[0];
    int A_next = A_share[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil_div(C_length,TILE_SIZE);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    while(counter < total_iteration){
      for(int i=0;i<TILE_SIZE;i+=blockDim.x){
        if (i+threadIdx.x<A_length-A_consumed){
          A_S[i+threadIdx.x]=A[A_curr+A_consumed+i+threadIdx.x];
        }
        else{
          A_S[i+threadIdx.x] = FLT_MAX;
        }
      }
      for (int i=0;i<TILE_SIZE;i+=blockDim.x){
        if(i+threadIdx.x<B_length-B_consumed){
          B_S[i+threadIdx.x]=B[B_curr+B_consumed+i+threadIdx.x];
        }
        else{
          B_S[i+threadIdx.x] = FLT_MAX;
        }
      }
      __syncthreads();

    int c_curr = threadIdx.x*(TILE_SIZE/blockDim.x);//ceil_div(TILE_SIZE,blockDim.x);
    int c_next = (threadIdx.x+1)*(TILE_SIZE/blockDim.x);
    c_curr = (c_curr<=C_length-C_completed)?c_curr:C_length-C_completed;
    c_next = (c_next<=C_length-C_completed)?c_next:C_length-C_completed;

    int a_curr = co_rank(c_curr,A_S,min(TILE_SIZE,A_length-A_consumed),B_S,min(TILE_SIZE,B_length-B_consumed));
    int b_curr = c_curr-a_curr;
    int a_next = co_rank(c_next,A_S,min(TILE_SIZE,A_length-A_consumed),B_S,min(TILE_SIZE,B_length-B_consumed));
    int b_next = c_next - a_next;

    merge_sequential(A_S+a_curr,a_next-a_curr,B_S+b_curr,b_next-b_curr,C+C_curr+C_completed+c_curr);
    counter ++;
    C_completed += TILE_SIZE;
    A_consumed += co_rank(TILE_SIZE,A_S,TILE_SIZE,B_S,TILE_SIZE);
    B_consumed = C_completed-A_consumed;
    __syncthreads();
    }
}
/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    __shared__ float shareAB[2*TILE_SIZE];
    float *A_S=&shareAB[0];
    float *B_S=&shareAB[TILE_SIZE];
    int C_curr = blockIdx.x * ceil_div(A_len+B_len,gridDim.x);
    int C_next = min((blockIdx.x+1)*ceil_div(A_len+B_len,gridDim.x),A_len+B_len);
    __shared__ int A_share[2];
    if (threadIdx.x==0){
      A_share[0] = co_rank(C_curr,A,A_len,B,B_len);
      A_share[1] = co_rank(C_next,A,A_len,B,B_len);
    }
    __syncthreads();

    int A_curr = A_share[0];
    int A_next = A_share[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil_div(C_length,TILE_SIZE);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = TILE_SIZE;
    int B_S_consumed = TILE_SIZE;
    while(counter < total_iteration){
      for(int i=0;i<A_S_consumed;i+=blockDim.x){
        if ((i+threadIdx.x<A_length-A_consumed)&&(i+threadIdx.x<A_S_consumed)){
          A_S[(A_S_start+TILE_SIZE-A_S_consumed+i+threadIdx.x)%TILE_SIZE]=A[A_curr+A_consumed+i+threadIdx.x];
        }
        //else{
        //  A_S[(A_S_start+TILE_SIZE-A_S_consumed+i+threadIdx.x)%TILE_SIZE] = FLT_MAX;
        //}
      }
      for (int i=0;i<B_S_consumed;i+=blockDim.x){
        if(i+threadIdx.x<B_length-B_consumed&&(i+threadIdx.x)<B_S_consumed){
          B_S[(B_S_start+TILE_SIZE-B_consumed+i+threadIdx.x)%TILE_SIZE]=B[B_curr+B_consumed+i+threadIdx.x];//Typo
        }
        //else{
        //  B_S[(B_S_start+TILE_SIZE-B_consumed+i+threadIdx.x)%TILE_SIZE] = FLT_MAX;
        //}
      }
      __syncthreads();

    int c_curr = threadIdx.x*(TILE_SIZE/blockDim.x);//ceil_div(TILE_SIZE,blockDim.x);
    int c_next = (threadIdx.x+1)*(TILE_SIZE/blockDim.x);
    c_curr = (c_curr<=C_length-C_completed)?c_curr:C_length-C_completed;
    c_next = (c_next<=C_length-C_completed)?c_next:C_length-C_completed;

    int a_curr = co_rank_circular(c_curr,A_S,min(TILE_SIZE,A_length-A_consumed),B_S,min(TILE_SIZE,B_length-B_consumed),A_S_start,B_S_start);
    int b_curr = c_curr-a_curr;
    int a_next = co_rank_circular(c_next,A_S,min(TILE_SIZE,A_length-A_consumed),B_S,min(TILE_SIZE,B_length-B_consumed),A_S_start,B_S_start);
    int b_next = c_next - a_next;

    merge_sequential_circular(A_S,a_next-a_curr,B_S,b_next-b_curr,C+C_curr+C_completed+c_curr,A_S_start+a_curr,B_S_start+b_curr);
    counter ++;
    A_S_consumed=co_rank_circular(min(TILE_SIZE,C_length-C_completed),A_S,min(TILE_SIZE,A_length-A_consumed),B_S,min(TILE_SIZE,B_length-B_consumed),A_S_start,B_S_start);
    B_S_consumed=min(TILE_SIZE,C_length-C_completed)-A_S_consumed;
    A_consumed+=A_S_consumed;
    C_completed += min(TILE_SIZE,C_length-C_completed);
    B_consumed=C_completed-A_consumed;
    A_S_start=A_S_start+A_S_consumed;
    if(A_S_start>=TILE_SIZE){
      A_S_start=A_S_start-TILE_SIZE;
    }
    B_S_start = B_S_start + B_S_consumed;
    if(B_S_start>=TILE_SIZE){
      B_S_start=B_S_start-TILE_SIZE;
    }
    __syncthreads();
    }
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
