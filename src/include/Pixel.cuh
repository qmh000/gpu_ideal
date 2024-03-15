#pragma once
#include "Point.cuh"

enum PartitionStatus{
	OUT = 0,
	BORDER = 1,
	IN = 2
};

struct EdgeSequence{
    int pos = 0;
    int len = 0;
};

class box{
public:
    double low[2];
	double high[2];

public:
    __host__ __device__ box();
    __host__ __device__ box (double lowx, double lowy, double highx, double highy){
        low[0] = lowx;
        low[1] = lowy;
        high[0] = highx;
        high[1] = highy;
    }
    
    ~box() = default;

    double get_lowx();
    double get_lowy();
    double get_highx();
    double get_highy();
    void set_box(double lowx, double lowy, double highx, double highy);
    
    //filtering
	__host__ __device__ bool contain(Point &p){
        return p.x>=low[0]&&
            p.x<=high[0]&&
            p.y>=low[1]&&
            p.y<=high[1];
    }

    __host__ __device__ bool contain(box &target){
        return target.low[0]>=low[0]&&
		       target.high[0]<=high[0]&&
		       target.low[1]>=low[1]&&
		       target.high[1]<=high[1];
    }
};

class Pixel{
public:
    int *status = nullptr;
    int *pointer = nullptr;
    int numPixels = 0;
    EdgeSequence *edge_sequences = nullptr;
    int totalLength = 0;

public:
    Pixel() = default;
    Pixel(int num_vertices);
    __host__ __device__ ~Pixel();
    void init_edge_sequences(int num_edge_seqs);
        
    //utility functions
    int get_numPixels();
    void add_edgeOffset(int id, int off);
    void add_edge(int idx, int start, int end);
    int get_totalLength();
    __host__ __device__ void set_status(int id, PartitionStatus status);
    __host__ __device__ PartitionStatus show_status(int id){
        int st = status[id];
        if(st == 0) return OUT;
        if(st == 2) return IN;
        return BORDER;
    }

    void process_null(int x, int y);
};