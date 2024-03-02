#pragma once
#include <stddef.h>

class Grid_line{
private:
	int *offset = nullptr;
	double *intersection_nodes = nullptr;

	int numCrosses = 0;
public:
	Grid_line() = default;
    Grid_line(int size);
	~Grid_line();
	void init_intersection_nodes(int num_nodes);

	//utility functions
	void set_offset(int gid, int off);
	int get_offset(int gid);
	void add_node(int idx, double x);
	double get_node(int idx);
};