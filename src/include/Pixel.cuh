#pragma once

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
private:
    double low[2];
	double high[2];

public:
    box();
    ~box() = default;
    box (double lowx, double lowy, double highx, double highy);

    double get_lowx();
    double get_lowy();
    double get_highx();
    double get_highy();
    void set_box(double lowx, double lowy, double highx, double highy);
};

class Pixel{
private:
    int *status = nullptr;
    int *pointer = nullptr;
    EdgeSequence *edge_sequences = nullptr;
    int totalLength = 0;

public:
    Pixel() = default;
    Pixel(int num_vertices);
    ~Pixel();
    void init_edge_sequences(int num_edge_seqs);
        
    //utility functions
    void add_edgeOffset(int id, int off);
    void add_edge(int idx, int start, int end);
    int get_totalLength();
    __host__ __device__ void set_status(int id, PartitionStatus status);
    __host__ __device__ PartitionStatus show_status(int id);

    void process_null(int x, int y);
};