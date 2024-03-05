#include "../include/Grid_line.cuh"
#include <cstring>

Grid_line::Grid_line(int size){
    gridNum = size + 2;
    offset = new int[size + 2];
    memset(offset, 0, sizeof(int) * (size+2));
}

__host__ __device__ Grid_line::~Grid_line(){
    if(offset) delete []offset;
    if(intersection_nodes) delete []intersection_nodes;
}

void Grid_line::init_intersection_nodes(int num_nodes){
    numCrosses = num_nodes;
    intersection_nodes = new double[num_nodes];
    
}

int Grid_line::get_offset(int gid){return offset[gid];}

void Grid_line::set_offset(int gid, int off){
    offset[gid] = off;
}

double Grid_line::get_node(int idx){return intersection_nodes[idx];}

void Grid_line::add_node(int idx, double x){
    intersection_nodes[idx] = x;
}

int Grid_line::get_num_nodes(int y){
    return offset[y + 1] - offset[y];
}
