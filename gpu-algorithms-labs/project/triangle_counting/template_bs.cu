#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array

  // Determine how many elements of those two arrays are common
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numEdges){
    uint32_t src_node = edgeSrc[tid];//u

    int dst_node = edgeDst[tid];//v
    int src = rowPtr[src_node];//u_ptr->P[u]
    int dst = rowPtr[dst_node];//v_ptr->P[v]
    int src_end = rowPtr[src_node+1];//u_end->P[u+1]
    int dst_end = rowPtr[dst_node+1];//v_end->P[v+1]
    int w1 = edgeDst[src], w2 = edgeDst[dst];//w1->Zd[u_ptr] w2->Zd[v_ptr]
    uint64_t num_triangle = 0;
    while (src < src_end && dst < dst_end){  //u_ptr<u_end && v_ptr<v_end
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



__global__ static void kernel_tc_binary_search(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array

  // Determine how many elements of those two arrays are common
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numEdges){
    uint32_t src_node = edgeSrc[tid];//u
    int dst_node = edgeDst[tid];//v
    int src = rowPtr[src_node];//u_ptr->P[u]
    int dst = rowPtr[dst_node];//v_ptr->P[v]
    int src_end = rowPtr[src_node+1];//u_end->P[u+1]
    int dst_end = rowPtr[dst_node+1];//v_end->P[v+1]
    //int w1 = edgeDst[src], w2 = edgeDst[dst];//w1->Zd[u_ptr] w2->Zd[v_ptr]
    uint64_t num_triangle = 0;
    int middle;
    //while (src < src_end && dst < dst_end){  //u_ptr<u_end && v_ptr<v_end
    for (int i=src;i<src_end;i++){
      int low = 0; 
      int high = (dst_end-dst)-1;
      while (high >= low){
        middle = (low+high)/2;
        if (edgeDst[dst+middle]==edgeDst[i]){
          num_triangle += 1;
          break;
        }
        else if(edgeDst[dst+middle] < edgeDst[i]){
          low = middle + 1;
        }
        else{
          high = middle - 1;
        }
      }
    }
    triangleCounts[tid] = num_triangle;
  }
}




uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // REQUIRED

    //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
    // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
    // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
    //@@ launch the linear search kernel here
    pangolin::Vector <uint64_t> edge_array (view.nnz(),0);
    dim3 dimBlock(512);
    dim3 dimGrid(ceil(float(view.nnz())/512),1,1);
    //pangolin::Vector<uint64_t>edge_array = pangolin::Vector<uint64_t>(view.nnz()) ;

    kernel_tc <<<dimGrid,dimBlock>>>(edge_array.data(), view.row_ind(),view.col_ind(),view.row_ptr(),view.nnz());
    cudaDeviceSynchronize();
    uint64_t total = 0;
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
    for (int i = 0; i< view.nnz(); i++){
      //total += triangle_Counts.data()[i];
      total += edge_array[i];
    }
    return total;

  } else if (2 == mode) {

    // OPTIONAL. See README for more details
    pangolin::Vector<uint64_t> edge_array(view.nnz(),0);
    dim3 dimBlock(512);
    dim3 dimGrid(ceil(float(view.nnz())/512),1,1);
    kernel_tc_binary_search<<<dimGrid,dimBlock>>>(edge_array.data(),view.row_ind(),view.col_ind(),view.row_ptr(),view.nnz());
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
