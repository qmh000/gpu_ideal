#pragma once

#include "util.h"
#include <cstddef>
#include "Pixel.cuh"
#include "MyRaster.cuh"
#include "query_context.cuh"

typedef struct PolygonMeta_{
	uint size; // size of the polygon in bytes
	uint num_vertices; // number of vertices in the boundary (hole excluded)
	size_t offset; // the offset in the file
	box mbr; // the bounding boxes
} PolygonMeta;

class MyPolygon{
public:
    size_t id = 0;
    box *mbr = nullptr;
    MyRaster *raster = nullptr;
    VertexSequence *boundary = nullptr;
    pthread_mutex_t ideal_partition_lock;
public:
    MyPolygon();
    ~MyPolygon();

    // some utility functions
    int get_numVertices();

    // for filtering
    box* getMBR();
    void rasterization(int vertex_per_raster);

    size_t decode(char *source);
     
};

void preprocess(query_context *gctx);

std::vector<MyPolygon *> load_binary_file(const char *path, query_context &global_ctx);
Point* load_points(const char *path, int &size);