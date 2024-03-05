#include "../include/helper_cuda.h"
#include "../include/MyPolygon.cuh"
#include <ctime>

#define BLOCK_SIZE 1024

__global__ void kernel(MyRaster* rasters1, MyRaster* rasters2, Point* points, int size, int *res1, int *res2){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < size){
        if(rasters1[x].contain(points[x]) != rasters2[x].contain(points[x])){
             printf("From GPU: %d\n", x);
        }
    }
}

int main(int argc, char** argv){
    query_context global_ctx;
    global_ctx.num_threads = 1;
    global_ctx.vpr = 10;
    global_ctx.source_polygons = load_binary_file("/home/qmh/data/child.idl", global_ctx);

    preprocess(&global_ctx);

    printf("Rasterization Finished!\n");

    // global_ctx.source_polygons[64021]->print(true);
    // global_ctx.source_polygons[64021]->raster->print();

    // int len = 0;
    // Point* points = load_points("/home/qmh/data/child_points.dat", len);
    // printf("(POINT(%lf %lf)\n", points[64021].x, points[64021].y);

    // if(global_ctx.source_polygons[64021]->raster->contain(points[64021])){
    //     std::cout << "Yes" << std::endl;
    // }else{
    //     std::cout << "No" << std::endl;
    // }

    // return 0;

    query_context global_ctx2;
    global_ctx2.num_threads = 1;
    global_ctx2.vpr = 20;
    global_ctx2.source_polygons = load_binary_file("/home/qmh/data/child.idl", global_ctx2);
    
    preprocess(&global_ctx2);

    printf("Rasterization Finished!\n");

    int size1 = global_ctx.source_polygons.size(), size2 = 0;

    MyRaster* h_rasters = new MyRaster[size1];
    MyRaster* h_rasters2 = new MyRaster[size1];
    Point* h_points = load_points("/home/qmh/data/child_points.dat", size2);

    // int size = 10;
    int size = min(size1, size2);
    MyRaster* d_rasters = nullptr;
    MyRaster* d_rasters2 = nullptr;
    Point* d_points = nullptr;
    int memsize1 = sizeof(MyRaster) * size;
    int memsize2 = sizeof(Point) * size;

    checkCudaErrors(cudaMalloc((void**) &d_rasters, memsize1));
    checkCudaErrors(cudaMalloc((void**) &d_rasters2, memsize1));
    checkCudaErrors(cudaMalloc((void**) &d_points, memsize2));

    checkCudaErrors(cudaMemcpy(d_points, h_points, memsize2, cudaMemcpyHostToDevice));

    for(int i = 0; i < size; i ++){
        h_rasters[i].dimx = global_ctx.source_polygons[i]->raster->dimx;
        h_rasters[i].dimy = global_ctx.source_polygons[i]->raster->dimy;
        h_rasters[i].step_x = global_ctx.source_polygons[i]->raster->step_x;
        h_rasters[i].step_y = global_ctx.source_polygons[i]->raster->step_y;

        h_rasters2[i].dimx = global_ctx2.source_polygons[i]->raster->dimx;
        h_rasters2[i].dimy = global_ctx2.source_polygons[i]->raster->dimy;
        h_rasters2[i].step_x = global_ctx2.source_polygons[i]->raster->step_x;
        h_rasters2[i].step_y = global_ctx2.source_polygons[i]->raster->step_y;

        // mbr
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].mbr, sizeof(box)));
        checkCudaErrors(cudaMemcpy(h_rasters[i].mbr, global_ctx.source_polygons[i]->raster->mbr, sizeof(box), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **) &h_rasters2[i].mbr, sizeof(box)));
        checkCudaErrors(cudaMemcpy(h_rasters2[i].mbr, global_ctx2.source_polygons[i]->raster->mbr, sizeof(box), cudaMemcpyHostToDevice));

        // vs
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].vs, sizeof(VertexSequence)));
        VertexSequence* h_vs = new VertexSequence();
        int vsNum = global_ctx.source_polygons[i]->get_numVertices();
        h_vs->numVertices = vsNum;
        checkCudaErrors(cudaMalloc((void **) &h_vs->p, vsNum * sizeof(Point)));
        checkCudaErrors(cudaMemcpy(h_vs->p, global_ctx.source_polygons[i]->raster->vs->p, vsNum * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters[i].vs, h_vs, sizeof(VertexSequence), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **) &h_rasters2[i].vs, sizeof(VertexSequence)));
        VertexSequence* h_vs2 = new VertexSequence();
        int vsNum2 = global_ctx2.source_polygons[i]->get_numVertices();
        h_vs2->numVertices = vsNum2;
        checkCudaErrors(cudaMalloc((void **) &h_vs2->p, vsNum2 * sizeof(Point)));
        checkCudaErrors(cudaMemcpy(h_vs2->p, global_ctx2.source_polygons[i]->raster->vs->p, vsNum2 * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters2[i].vs, h_vs2, sizeof(VertexSequence), cudaMemcpyHostToDevice));

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

        checkCudaErrors(cudaMalloc((void **) &h_rasters2[i].pixels, sizeof(Pixel)));
        Pixel *h_pixels2 = new Pixel();
        int pixNum2 = global_ctx2.source_polygons[i]->raster->pixels->get_numPixels();
        h_pixels2->numPixels = pixNum2;
        checkCudaErrors(cudaMalloc((void **) &h_pixels2->status, pixNum2 * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &h_pixels2->pointer, (pixNum2+1) * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_pixels2->status, global_ctx2.source_polygons[i]->raster->pixels->status, pixNum2 * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_pixels2->pointer, global_ctx2.source_polygons[i]->raster->pixels->pointer, (pixNum2+1) * sizeof(int), cudaMemcpyHostToDevice));
        int edgeSeqNum2 = global_ctx2.source_polygons[i]->raster->pixels->totalLength;
        h_pixels2->totalLength = edgeSeqNum2;
        checkCudaErrors(cudaMalloc((void **) &h_pixels2->edge_sequences, edgeSeqNum2 * sizeof(EdgeSequence)));
        checkCudaErrors(cudaMemcpy(h_pixels2->edge_sequences, global_ctx2.source_polygons[i]->raster->pixels->edge_sequences, edgeSeqNum2 * sizeof(EdgeSequence), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters2[i].pixels, h_pixels2, sizeof(Pixel), cudaMemcpyHostToDevice));

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

        checkCudaErrors(cudaMalloc((void **) &h_rasters2[i].horizontal, sizeof(Grid_line)));
        Grid_line *h_horizontal2 = new Grid_line();
        int gridNum12 = global_ctx2.source_polygons[i]->raster->horizontal->gridNum;
        h_horizontal2->gridNum = gridNum12 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_horizontal2->offset, gridNum12 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_horizontal2->offset, global_ctx2.source_polygons[i]->raster->horizontal->offset, gridNum12 * sizeof(int), cudaMemcpyHostToDevice));
        int numCrosses12 = global_ctx2.source_polygons[i]->raster->horizontal->numCrosses;
        h_horizontal2->numCrosses = numCrosses12;
        checkCudaErrors(cudaMalloc((void **) &h_horizontal2->intersection_nodes, numCrosses12 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_horizontal2->intersection_nodes, global_ctx2.source_polygons[i]->raster->horizontal->intersection_nodes, numCrosses12 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters2[i].horizontal, h_horizontal2, sizeof(Grid_line), cudaMemcpyHostToDevice));
        
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

        checkCudaErrors(cudaMalloc((void **) &h_rasters2[i].vertical, sizeof(Grid_line)));
        Grid_line *h_vertical2 = new Grid_line();
        int gridNum22 = global_ctx2.source_polygons[i]->raster->vertical->gridNum;
        h_vertical2->gridNum = gridNum22 + 1;
        checkCudaErrors(cudaMalloc((void **) &h_vertical2->offset, gridNum22 * sizeof(int)));
        checkCudaErrors(cudaMemcpy(h_vertical2->offset, global_ctx2.source_polygons[i]->raster->vertical->offset, gridNum22 * sizeof(int), cudaMemcpyHostToDevice));
        int numCrosses22 =  global_ctx2.source_polygons[i]->raster->vertical->numCrosses;
        h_vertical2->numCrosses = numCrosses22;
        checkCudaErrors(cudaMalloc((void **) &h_vertical2->intersection_nodes, numCrosses22 * sizeof(double)));
        checkCudaErrors(cudaMemcpy(h_vertical2->intersection_nodes, global_ctx2.source_polygons[i]->raster->vertical->intersection_nodes, numCrosses22 * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters2[i].vertical, h_vertical2, sizeof(Grid_line), cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMemcpy(d_rasters, h_rasters, memsize1, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rasters2, h_rasters2, memsize1, cudaMemcpyHostToDevice));

    int *d_result1 = NULL;
	checkCudaErrors(cudaMalloc((void**) &d_result1, sizeof(int)));
	checkCudaErrors(cudaMemset(d_result1, 0, sizeof(int)));

    int *d_result2 = NULL;
	checkCudaErrors(cudaMalloc((void**) &d_result2, sizeof(int)));
	checkCudaErrors(cudaMemset(d_result2, 0, sizeof(int)));

    const int grid_size_x = ceil(size / static_cast<float>(BLOCK_SIZE));
    const dim3 block_size(BLOCK_SIZE, 1, 1);
    const dim3 grid_size(grid_size_x, 1, 1);

    timeval start = get_cur_time();

    kernel<<<grid_size, block_size>>>(d_rasters, d_rasters2, d_points, size, d_result1, d_result2);
    cudaDeviceSynchronize();

    logt("GPU process", start);

    int h_result1;
    int h_result2;
	checkCudaErrors(cudaMemcpy(&h_result1, d_result1, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_result2, d_result2, sizeof(int), cudaMemcpyDeviceToHost));

    printf("EPP = 10: %d\n", h_result1);
    printf("EPP = 20: %d\n", h_result2);



}