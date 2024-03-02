#pragma once
#include "vertex_sequence.cuh"
#include "Grid_line.cuh"
#include "Pixel.cuh"
#include <map>
#include <vector>
#include <string>

enum cross_type{
	ENTER = 0,
	LEAVE = 1
};

class cross_info{
public:
	cross_type type;
	int edge_id;
	cross_info(cross_type t, int e){
		type = t;
		edge_id = e;
	}
};

class MyRaster{
public:
    double step_x = 0.0;
	double step_y = 0.0;
	int dimx = 0;
	int dimy = 0;
    
    box *mbr = nullptr;
    VertexSequence *vs = nullptr;
    Pixel *pixels = nullptr;
    Grid_line *horizontal;
	Grid_line *vertical;

public:
    MyRaster() = default;
    MyRaster(VertexSequence *vs, int epp);
    ~MyRaster();

    // rasterization
    void rasterization();
    void init_pixels();
	void evaluate_edges();
	void scanline_reandering();
    void process_crosses(std::map<int, std::vector<cross_info>> edges_info);
    void process_intersection(std::map<int, std::vector<double>> intersection_info, std::string direction);
    void process_pixels(int x, int y);

    // operate pixels
    int get_id(int x, int y);
    int get_x(int id);
    int get_y(int id);

    //utility functions
    box* get_mbr();
    void set_mbr(box* addr);
    VertexSequence* get_vs();
    void set_vs(VertexSequence* addr);
    Pixel* get_pix();
    void set_pix(Pixel* addr);
    Grid_line* get_horizontal();
    void set_horizontal(Grid_line* addr);
    Grid_line* get_vertical();
    void set_vertical(Grid_line* addr);
};