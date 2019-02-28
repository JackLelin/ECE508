#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the global queue
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int node = currLevelNodes[tid];
  if (tid < *numCurrLevelNodes){ 
    for (unsigned int nbrIdx  = nodePtrs[node]; nbrIdx < nodePtrs[node+1];nbrIdx++){
      unsigned int NeighborId  = nodeNeighbors[nbrIdx];
      if (atomicAdd(&nodeVisited[NeighborId],+1)==0){   
        int position = atomicAdd(numNextLevelNodes,1);
        nextLevelNodes[position] = NeighborId;
      } 
    } 
  }   
}
__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the block queue
  // If full, add it to the global queue

  // Calculate space for block queue to go into global queue

  // Store block queue in global queue
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int queue_size;
  __shared__ int queue[BQ_CAPACITY];
  if (tid < BQ_CAPACITY){
     queue[tid] = 0;
  }
  __syncthreads();
  queue_size = 0;
  __syncthreads();
  if (tid < *numCurrLevelNodes){
    unsigned int node = currLevelNodes[tid]; 
    for (unsigned int nbrIdx = nodePtrs[node];nbrIdx<nodePtrs[node+1];nbrIdx++){
      unsigned int NeighborId = nodeNeighbors[nbrIdx];
      if (atomicAdd(&nodeVisited[NeighborId],+1)==0){
         int temp = atomicAdd(&queue_size,1);
         if (temp <BQ_CAPACITY){
          
            queue[temp] = NeighborId;
         }
         else{
             int position = atomicAdd(numNextLevelNodes,1);
             nextLevelNodes[position] = NeighborId;
         }
      }
    }
  }
  __syncthreads();
  __shared__ int global_offset;
  if (threadIdx.x==0){
    global_offset = atomicAdd(numNextLevelNodes,queue_size);
  }
  __syncthreads();
  for (int i=threadIdx.x;i < queue_size; i+=blockDim.x){
     nextLevelNodes[global_offset+i] = queue[i];
    
  }
  __syncthreads();
}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses one queue per warp

  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the warp queue
  // If full, add it to the block queue
  // If full, add it to the global queue

  // Calculate space for warp queue to go into block queue

  // Store warp queue in block queue
  // If full, add it to the global queue

  // Calculate space for block queue to go into global queue
  // Saturate block queue counter
  // Calculate space for global queue

  // Store block queue in global queue
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int w_id = threadIdx.x / WARP_SIZE;
//  int w_off = tid % WARP_SIZE;

  __shared__ int b_queue_size;
  __shared__ int b_queue[BQ_CAPACITY];
  __shared__ int w_queue[NUM_WARPS][WQ_CAPACITY];//[WQ_CAPACITY];
  __shared__ int w_queue_size[NUM_WARPS];
  for (int i =0; i<NUM_WARPS;i++){
    w_queue_size[i] = 0;
  }
  __syncthreads();
  b_queue_size = 0;
  __syncthreads();
  if (tid < *numCurrLevelNodes){
    unsigned int node = currLevelNodes[tid];
    for (unsigned int nbrIdx = nodePtrs[node];nbrIdx<nodePtrs[node+1];nbrIdx++){
      unsigned int NeighborId = nodeNeighbors[nbrIdx];
      if (atomicAdd(&nodeVisited[NeighborId],+1)==0){
         int temp_w = atomicAdd(&w_queue_size[w_id],1);
         if (temp_w <WQ_CAPACITY){
 
            w_queue[w_id][temp_w] = NeighborId;
         }
         if (temp_w >= WQ_CAPACITY){
	   int b_position = atomicAdd(&b_queue_size,1);
           if (b_position < BQ_CAPACITY){
             b_queue[b_position] = NeighborId;
           }
           else{
             int position = atomicAdd(numNextLevelNodes,1);
             nextLevelNodes[position] = NeighborId;
           }
         }
      }
    }
  }

  __syncthreads();
  __shared__ int block_offset[NUM_WARPS];
  if (threadIdx.x < NUM_WARPS){
    block_offset[threadIdx.x] = atomicAdd(&b_queue_size,w_queue_size[threadIdx.x]);
  }
  __syncthreads();
  
  for (int i=0;i<NUM_WARPS;i++){
    if (threadIdx.x <w_queue_size[i]){
      if (threadIdx.x + block_offset[i]<BQ_CAPACITY){
        b_queue[threadIdx.x+block_offset[i]] = w_queue[i][threadIdx.x];
      }
      else{
        int temp = atomicAdd(numNextLevelNodes,1);
        nextLevelNodes[temp] = w_queue[i][threadIdx.x];  
      }
    }
  }
  __syncthreads();

  __shared__ int global_offset;
  if (threadIdx.x==0){
    global_offset = atomicAdd(numNextLevelNodes,b_queue_size);
  }
  __syncthreads();

  for (int i=threadIdx.x;i < b_queue_size; i+=blockDim.x){
     nextLevelNodes[global_offset+i] = b_queue[i];
    
  }
  __syncthreads();
}



/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
