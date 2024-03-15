#include "../include/helper_cuda.h"
#include "../include/MyPolygon.cuh"
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define BLOCK_SIZE 32

__global__ void kernel(MyRaster* source, MyRaster* target, int size1, int size2, int *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < size1 && y < size2){
        if(source[x].contain(target[y])){
            atomicAdd(result, 1);
        }
    }
}

int main(int argc, char** argv){
    query_context global_ctx;
    // global_ctx.num_threads = 1;
    std::cout << "EPP = ";
    std::cin >> global_ctx.vpr;
    // global_ctx.vpr = 100;
    global_ctx.source_polygons = load_binary_file("/home/qmh/data/child.idl", global_ctx);
    global_ctx.target_polygons = load_binary_file("/home/qmh/data/has_child.idl", global_ctx);
    global_ctx.target_num = global_ctx.target_polygons.size();
    
    preprocess(&global_ctx);

    printf("Rasterization Finished! EPP = %d\n", global_ctx.vpr);

    int size1 = global_ctx.source_polygons.size(), size2 = global_ctx.target_polygons.size();
    // size1 = size2;
    // int size = min(size1, size2);

    MyRaster* h_sourceRasters = new MyRaster[size1];
    MyRaster* h_targetRasters = new MyRaster[size2];

    
    MyRaster* d_sourceRasters = nullptr;
    MyRaster* d_targetRasters = nullptr;
    int memsize1 = sizeof(MyRaster) * size1;
    int memsize2 = sizeof(MyRaster) * size2;

    checkCudaErrors(cudaMalloc((void**) &d_sourceRasters, memsize1));
    checkCudaErrors(cudaMalloc((void**) &d_targetRasters, memsize2));

    for(int i = 0; i < size1; i ++){
        h_sourceRasters[i].dimx = global_ctx.source_polygons[i]->raster->dimx;
        h_sourceRasters[i].dimy = global_ctx.source_polygons[i]->raster->dimy;
        h_sourceRasters[i].step_x = global_ctx.source_polygons[i]->raster->step_x;
        h_sourceRasters[i].step_y = global_ctx.source_polygons[i]->raster->step_y;

        // mbr
        checkCudaErrors(cudaMalloc((void **) &h_sourceRasters[i].mbr, sizeof(box)));
        checkCudaErrors(cudaMemcpy(h_sourceRasters[i].mbr, global_ctx.source_polygons[i]->raster->mbr, sizeof(box), cudaMemcpyHostToDevice));

        // vs
        checkCudaErrors(cudaMalloc((void **) &h_sourceRasters[i].vs, sizeof(VertexSequence)));
        VertexSequence* h_sourceVs = new VertexSequence();
        int vsNum1 = global_ctx.source_polygons[i]->get_numVertices();
        h_sourceVs->numVertices = vsNum1;
        checkCudaErrors(cudaMalloc((void **) &h_sourceVs->p, vsNum1 * sizeof(Point)));
        checkCudaErrors(cudaMemcpy(h_sourceVs->p, global_ctx.source_polygons[i]->raster->vs->p, vsNum1 * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_sourceRasters[i].vs, h_sourceVs, sizeof(VertexSequence), cudaMemcpyHostToDevice));

        // pixels
        checkCudaErrors(cudaMalloc((void **) &h_sourceRasters[i].pixels, sizeof(Pixel)));
        Pixel *h_sourcePixels = new Pixel();
        int pixNum1 = global_ctx.source_polygons[i]->raster->pixels->get_numPixels();
        h_sourcePixels->numPixels = pixNum1;
        checkCudaErrors(cudaMalloc((void **) &h_sourcePixels->status, pixNum1 * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &h_sourcePixels->pointer, (pixNum1+1) * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_sourcePixels->status, global_ctx.source_polygons[i]->raster->pixels->status, pixNum1 * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_sourcePixels->pointer, global_ctx.source_polygons[i]->raster->pixels->pointer, (pixNum1+1) * sizeof(int), cudaMemcpyHostToDevice));
        int edgeSeqNum1 = global_ctx.source_polygons[i]->raster->pixels->totalLength;
        h_sourcePixels->totalLength = edgeSeqNum1;
        checkCudaErrors(cudaMalloc((void **) &h_sourcePixels->edge_sequences, edgeSeqNum1 * sizeof(EdgeSequence)));
        checkCudaErrors(cudaMemcpy(h_sourcePixels->edge_sequences, global_ctx.source_polygons[i]->raster->pixels->edge_sequences, edgeSeqNum1 * sizeof(EdgeSequence), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_sourceRasters[i].pixels, h_sourcePixels, sizeof(Pixel), cudaMemcpyHostToDevice));

        // horizaontal
        checkCudaErrors(cudaMalloc((void **) &h_sourceRasters[i].horizontal, sizeof(Grid_line)));
        Grid_line *h_sourceHorizontal = new Grid_line();
        int horizontal_gridNum1 = global_ctx.source_polygons[i]->raster->horizontal->gridNum;
        h_sourceHorizontal->gridNum = horizontal_gridNum1 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_sourceHorizontal->offset, horizontal_gridNum1 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_sourceHorizontal->offset, global_ctx.source_polygons[i]->raster->horizontal->offset, horizontal_gridNum1 * sizeof(int), cudaMemcpyHostToDevice));
        int horizontal_numCrosses1 =  global_ctx.source_polygons[i]->raster->horizontal->numCrosses;
        h_sourceHorizontal->numCrosses = horizontal_numCrosses1;
        checkCudaErrors(cudaMalloc((void **) &h_sourceHorizontal->intersection_nodes, horizontal_numCrosses1 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_sourceHorizontal->intersection_nodes, global_ctx.source_polygons[i]->raster->horizontal->intersection_nodes, horizontal_numCrosses1 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_sourceRasters[i].horizontal, h_sourceHorizontal, sizeof(Grid_line), cudaMemcpyHostToDevice));
        
        // vertical
        checkCudaErrors(cudaMalloc((void **) &h_sourceRasters[i].vertical, sizeof(Grid_line)));
        Grid_line *h_sourceVertical = new Grid_line();
        int vertical_gridNum1 = global_ctx.source_polygons[i]->raster->vertical->gridNum;
        h_sourceVertical->gridNum = vertical_gridNum1 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_sourceVertical->offset, vertical_gridNum1 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_sourceVertical->offset, global_ctx.source_polygons[i]->raster->vertical->offset, vertical_gridNum1 * sizeof(int), cudaMemcpyHostToDevice));
        int vertical_numCrosses1 =  global_ctx.source_polygons[i]->raster->vertical->numCrosses;
        h_sourceVertical->numCrosses = vertical_numCrosses1;
        checkCudaErrors(cudaMalloc((void **) &h_sourceVertical->intersection_nodes, vertical_numCrosses1 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_sourceVertical->intersection_nodes, global_ctx.source_polygons[i]->raster->vertical->intersection_nodes, vertical_numCrosses1 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_sourceRasters[i].vertical, h_sourceVertical, sizeof(Grid_line), cudaMemcpyHostToDevice));

    }

    for(int i = 0; i < size2; i ++){
        h_targetRasters[i].dimx = global_ctx.target_polygons[i]->raster->dimx;
        h_targetRasters[i].dimy = global_ctx.target_polygons[i]->raster->dimy;
        h_targetRasters[i].step_x = global_ctx.target_polygons[i]->raster->step_x;
        h_targetRasters[i].step_y = global_ctx.target_polygons[i]->raster->step_y;

        // mbr
        checkCudaErrors(cudaMalloc((void **) &h_targetRasters[i].mbr, sizeof(box)));
        checkCudaErrors(cudaMemcpy(h_targetRasters[i].mbr, global_ctx.target_polygons[i]->raster->mbr, sizeof(box), cudaMemcpyHostToDevice));

        // vs
        checkCudaErrors(cudaMalloc((void **) &h_targetRasters[i].vs, sizeof(VertexSequence)));
        VertexSequence* h_targetVs = new VertexSequence();
        int vsNum2 = global_ctx.target_polygons[i]->get_numVertices();
        h_targetVs->numVertices = vsNum2;
        checkCudaErrors(cudaMalloc((void **) &h_targetVs->p, vsNum2 * sizeof(Point)));
        checkCudaErrors(cudaMemcpy(h_targetVs->p, global_ctx.target_polygons[i]->raster->vs->p, vsNum2 * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_targetRasters[i].vs, h_targetVs, sizeof(VertexSequence), cudaMemcpyHostToDevice));\

        // pixel
        checkCudaErrors(cudaMalloc((void **) &h_targetRasters[i].pixels, sizeof(Pixel)));
        Pixel *h_targetPixels = new Pixel();
        int pixNum2 = global_ctx.target_polygons[i]->raster->pixels->get_numPixels();
        h_targetPixels->numPixels = pixNum2;
        checkCudaErrors(cudaMalloc((void **) &h_targetPixels->status, pixNum2 * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &h_targetPixels->pointer, (pixNum2+1) * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_targetPixels->status, global_ctx.target_polygons[i]->raster->pixels->status, pixNum2 * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_targetPixels->pointer, global_ctx.target_polygons[i]->raster->pixels->pointer, (pixNum2+1) * sizeof(int), cudaMemcpyHostToDevice));
        int edgeSeqNum2 = global_ctx.target_polygons[i]->raster->pixels->totalLength;
        h_targetPixels->totalLength = edgeSeqNum2;
        checkCudaErrors(cudaMalloc((void **) &h_targetPixels->edge_sequences, edgeSeqNum2 * sizeof(EdgeSequence)));
        checkCudaErrors(cudaMemcpy(h_targetPixels->edge_sequences, global_ctx.target_polygons[i]->raster->pixels->edge_sequences, edgeSeqNum2 * sizeof(EdgeSequence), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_targetRasters[i].pixels, h_targetPixels, sizeof(Pixel), cudaMemcpyHostToDevice));

        // horizontal
        checkCudaErrors(cudaMalloc((void **) &h_targetRasters[i].horizontal, sizeof(Grid_line)));
        Grid_line *h_targetHorizontal = new Grid_line();
        int horizontal_gridNum2 = global_ctx.target_polygons[i]->raster->horizontal->gridNum;
        h_targetHorizontal->gridNum = horizontal_gridNum2 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_targetHorizontal->offset, horizontal_gridNum2 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_targetHorizontal->offset, global_ctx.target_polygons[i]->raster->horizontal->offset, horizontal_gridNum2 * sizeof(int), cudaMemcpyHostToDevice));
        int horizontal_numCrosses2 =  global_ctx.target_polygons[i]->raster->horizontal->numCrosses;
        h_targetHorizontal->numCrosses = horizontal_numCrosses2;
        checkCudaErrors(cudaMalloc((void **) &h_targetHorizontal->intersection_nodes, horizontal_numCrosses2 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_targetHorizontal->intersection_nodes, global_ctx.target_polygons[i]->raster->horizontal->intersection_nodes, horizontal_numCrosses2 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_targetRasters[i].horizontal, h_targetHorizontal, sizeof(Grid_line), cudaMemcpyHostToDevice));

        // vertical
        checkCudaErrors(cudaMalloc((void **) &h_targetRasters[i].vertical, sizeof(Grid_line)));
        Grid_line *h_targetVertical = new Grid_line();
        int vertical_gridNum2 = global_ctx.target_polygons[i]->raster->vertical->gridNum;
        h_targetVertical->gridNum = vertical_gridNum2 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_targetVertical->offset, vertical_gridNum2 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_targetVertical->offset, global_ctx.target_polygons[i]->raster->vertical->offset, vertical_gridNum2 * sizeof(int), cudaMemcpyHostToDevice));
        int vertical_numCrosses2 =  global_ctx.target_polygons[i]->raster->vertical->numCrosses;
        h_targetVertical->numCrosses = vertical_numCrosses2;
        checkCudaErrors(cudaMalloc((void **) &h_targetVertical->intersection_nodes, vertical_numCrosses2 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_targetVertical->intersection_nodes, global_ctx.target_polygons[i]->raster->vertical->intersection_nodes, vertical_numCrosses2 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_targetRasters[i].vertical, h_targetVertical, sizeof(Grid_line), cudaMemcpyHostToDevice));

    }

    checkCudaErrors(cudaMemcpy(d_sourceRasters, h_sourceRasters, memsize1, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_targetRasters, h_targetRasters, memsize2, cudaMemcpyHostToDevice));

    int *d_result = nullptr;
	checkCudaErrors(cudaMalloc((void**) &d_result, sizeof(int)));
	checkCudaErrors(cudaMemset(d_result, 0, sizeof(int)));

    const int grid_size_x = ceil(size1 / static_cast<float>(BLOCK_SIZE));
    const int grid_size_y = ceil(size2 / static_cast<float>(BLOCK_SIZE));
    
    const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    timeval start = get_cur_time();

    kernel<<<grid_size, block_size>>>(d_sourceRasters, d_targetRasters, size1, size2, d_result);
    cudaDeviceSynchronize();

    logt("GPU process", start);

    int h_result;
	checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    printf("GPU: %d\n", h_result);

}