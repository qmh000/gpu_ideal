#include "../include/helper_cuda.h"
#include "../include/MyPolygon.cuh"

int main(int argc, char** argv){
    query_context global_ctx;
    global_ctx.num_threads = 1;
    global_ctx.source_polygons = load_binary_file("/home/qmh/data/has_child.idl", global_ctx);
    
    //析构函数有问题
    preprocess(&global_ctx);

    printf("Rasterization Finished!\n");



    int size1 = global_ctx.source_polygons.size(), size2 = 0;
    MyRaster* h_rasters = new MyRaster[size1];
    Point* h_points = load_points("/home/qmh/data/sampled.points.dat", size2);

    int size = min(size1, size2);
    MyRaster* d_rasters = nullptr;
    Point* d_points = nullptr;
    int memsize1 = sizeof(MyRaster) * size;
    int memsize2 = sizeof(Point) * size;

    checkCudaErrors(cudaMalloc((void**) &d_rasters, memsize1));
    checkCudaErrors(cudaMalloc((void**) &d_points, memsize2));

    checkCudaErrors(cudaMemcpy(d_points, h_points, memsize2, cudaMemcpyHostToDevice));

    for(int i = 0; i < size; i ++){
        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].mbr, sizeof(box)));
        checkCudaErrors(cudaMemcpy(h_rasters[i].mbr, global_ctx.source_polygons[i]->raster->mbr, sizeof(box), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].vs, sizeof(VertexSequence)));
        VertexSequence* h_vs = new VertexSequence();
        int vsNum = global_ctx.source_polygons[i]->get_numVertices();
        h_vs->numVertices = vsNum;
        checkCudaErrors(cudaMalloc((void **) &h_vs->p, vsNum * sizeof(Point)));
        checkCudaErrors(cudaMemcpy(h_vs, global_ctx.source_polygons[i]->raster->vs->p, vsNum * sizeof(Point), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(h_rasters[i].vs, h_vs, sizeof(VertexSequence), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **) &h_rasters[i].pixels, sizeof(Pixel)));
        
    }

    




}