#pragma once
#include <stddef.h>

class Grid_line{
public:
	int *offset = nullptr;

	int gridNum = 0;

	double *intersection_nodes = nullptr;

	int numCrosses = 0;
public:
	Grid_line() = default;
    Grid_line(int size);
	__host__ __device__ ~Grid_line();
	void init_intersection_nodes(int num_nodes);

	//utility functions
	void set_offset(int gid, int off);
	int get_offset(int gid);
	void add_node(int idx, double x);
	double get_node(int idx);
};