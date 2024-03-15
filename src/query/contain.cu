#include "../include/helper_cuda.h"
#include "../include/MyPolygon.cuh"
#include <ctime>

#define BLOCK_SIZE 1024

__global__ void kernel(MyRaster* rasters, Point* points, int size, int *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < size){
        if(rasters[x].contain(points[x])){
             atomicAdd(result, 1);
        }
    }
}

int main(int argc, char** argv){
    query_context global_ctx;
    global_ctx.num_threads = 1;
    std::cout << "EPP = ";
    std::cin >> global_ctx.vpr;
    // global_ctx.vpr = 100;
    global_ctx.source_polygons = load_binary_file("/home/qmh/data/child.idl", global_ctx);
    
    preprocess(&global_ctx);

    printf("Rasterization Finished! EPP = %d\n", global_ctx.vpr);

    int size1 = global_ctx.source_polygons.size(), size2 = 0;

    MyRaster* h_rasters = new MyRaster[size1];
    Point* h_points = load_points("/home/qmh/data/child_points.dat", size2);

    // int size = 10;
    int size = min(size1, size2);
    MyRaster* d_rasters = nullptr;
    Point* d_points = nullptr;
    int memsize1 = sizeof(MyRaster) * size;
    int memsize2 = sizeof(Point) * size;

    checkCudaErrors(cudaMalloc((void**) &d_rasters, memsize1));
    checkCudaErrors(cudaMalloc((void**) &d_points, memsize2));

    checkCudaErrors(cudaMemcpy(d_points, h_points, memsize2, cudaMemcpyHostToDevice));

    for(int i = 0; i < size; i ++){
        h_rasters[i].dimx = global_ctx.source_polygons[i]->raster->dimx;
        h_rasters[i].dimy = global_ctx.source_polygons[i]->raster->dimy;
        h_rasters[i].step_x = global_ctx.source_polygons[i]->raster->step_x;
        h_rasters[i].step_y = global_ctx.source_polygons[i]->raster->step_y;

        // mbr
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].mbr, sizeof(box)));
        checkCudaErrors(cudaMemcpy(h_rasters[i].mbr, global_ctx.source_polygons[i]->raster->mbr, sizeof(box), cudaMemcpyHostToDevice));

        // vs
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].vs, sizeof(VertexSequence)));
        VertexSequence* h_vs = new VertexSequence();
        int vsNum = global_ctx.source_polygons[i]->get_numVertices();
        h_vs->numVertices = vsNum;
        checkCudaErrors(cudaMalloc((void **) &h_vs->p, vsNum * sizeof(Point)));
        checkCudaErrors(cudaMemcpy(h_vs->p, global_ctx.source_polygons[i]->raster->vs->p, vsNum * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters[i].vs, h_vs, sizeof(VertexSequence), cudaMemcpyHostToDevice));

        // pixels
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].pixels, sizeof(Pixel)));
        Pixel *h_pixels = new Pixel();
        int pixNum = global_ctx.source_polygons[i]->raster->pixels->get_numPixels();
        h_pixels->numPixels = pixNum;
        checkCudaErrors(cudaMalloc((void **) &h_pixels->status, pixNum * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &h_pixels->pointer, (pixNum+1) * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_pixels->status, global_ctx.source_polygons[i]->raster->pixels->status, pixNum * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_pixels->pointer, global_ctx.source_polygons[i]->raster->pixels->pointer, (pixNum+1) * sizeof(int), cudaMemcpyHostToDevice));
        int edgeSeqNum = global_ctx.source_polygons[i]->raster->pixels->totalLength;
        h_pixels->totalLength = edgeSeqNum;
        checkCudaErrors(cudaMalloc((void **) &h_pixels->edge_sequences, edgeSeqNum * sizeof(EdgeSequence)));
        checkCudaErrors(cudaMemcpy(h_pixels->edge_sequences, global_ctx.source_polygons[i]->raster->pixels->edge_sequences, edgeSeqNum * sizeof(EdgeSequence), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters[i].pixels, h_pixels, sizeof(Pixel), cudaMemcpyHostToDevice));

        // horizaontal
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].horizontal, sizeof(Grid_line)));
        Grid_line *h_horizontal = new Grid_line();
        int gridNum1 = global_ctx.source_polygons[i]->raster->horizontal->gridNum;
        h_horizontal->gridNum = gridNum1 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_horizontal->offset, gridNum1 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_horizontal->offset, global_ctx.source_polygons[i]->raster->horizontal->offset, gridNum1 * sizeof(int), cudaMemcpyHostToDevice));
        int numCrosses1 =  global_ctx.source_polygons[i]->raster->horizontal->numCrosses;
        h_horizontal->numCrosses = numCrosses1;
        checkCudaErrors(cudaMalloc((void **) &h_horizontal->intersection_nodes, numCrosses1 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_horizontal->intersection_nodes, global_ctx.source_polygons[i]->raster->horizontal->intersection_nodes, numCrosses1 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters[i].horizontal, h_horizontal, sizeof(Grid_line), cudaMemcpyHostToDevice));
        
        // vertical
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].vertical, sizeof(Grid_line)));
        Grid_line *h_vertical = new Grid_line();
        int gridNum2 = global_ctx.source_polygons[i]->raster->vertical->gridNum;
        h_vertical->gridNum = gridNum2 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_vertical->offset, gridNum2 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_vertical->offset, global_ctx.source_polygons[i]->raster->vertical->offset, gridNum2 * sizeof(int), cudaMemcpyHostToDevice));
        int numCrosses2 =  global_ctx.source_polygons[i]->raster->vertical->numCrosses;
        h_vertical->numCrosses = numCrosses2;
        checkCudaErrors(cudaMalloc((void **) &h_vertical->intersection_nodes, numCrosses2 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_vertical->intersection_nodes, global_ctx.source_polygons[i]->raster->vertical->intersection_nodes, numCrosses2 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters[i].vertical, h_vertical, sizeof(Grid_line), cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMemcpy(d_rasters, h_rasters, memsize1, cudaMemcpyHostToDevice));

    int *d_result = nullptr;
	checkCudaErrors(cudaMalloc((void**) &d_result, sizeof(int)));
	checkCudaErrors(cudaMemset(d_result, 0, sizeof(int)));

    const int grid_size_x = ceil(size / static_cast<float>(BLOCK_SIZE));
    const dim3 block_size(BLOCK_SIZE, 1, 1);
    const dim3 grid_size(grid_size_x, 1, 1);

    timeval start = get_cur_time();

    kernel<<<grid_size, block_size>>>(d_rasters, d_points, size, d_result);
    cudaDeviceSynchronize();

    logt("GPU process", start);

    int h_result;
	checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    printf("GPU: %d\n", h_result);


    




}