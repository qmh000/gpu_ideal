#pragma once
#include "vertex_sequence.cuh"
#include "Grid_line.cuh"
#include "Pixel.cuh"
#include "util.h"
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
    __host__ __device__ ~MyRaster();

    // rasterization
    void rasterization();
    void init_pixels();
	void evaluate_edges();
	void scanline_reandering();
    void process_crosses(std::map<int, std::vector<cross_info>> edges_info);
    void process_intersection(std::map<int, std::vector<double>> intersection_info, std::string direction);
    void process_pixels(int x, int y);

    // operate pixels
    __host__ __device__ int get_id(int x, int y){
        assert(x>=0&&x<=dimx);
        assert(y>=0&&y<=dimy);
        return y * (dimx+1) + x;
    }
    __host__ __device__ int get_x(int id){
        return id % (dimx+1);
    }
    __host__ __device__ int get_y(int id){
        assert((id / (dimx+1)) <= dimy);
        return id / (dimx+1);
    }

    //utility functions
    void print();
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
    __host__ __device__ int get_offset_x(double xval){
        assert(mbr);
        assert(step_x>0.000000000001 && "the width per pixel must not be 0");
        int x = (int)((xval-mbr->low[0])/step_x);
        return min(max(x, 0), dimx);
    }
    // the range must be [0, dimy]
    __host__ __device__ int get_offset_y(double yval){
        assert(mbr);
        assert(step_y>0.000000000001 && "the hight per pixel must not be 0");
        int y = (int)((yval-mbr->low[1])/step_y);
        return min(max(y, 0), dimy);
    }
    __host__ __device__ int get_pixel_id(Point &p){
        int xoff = get_offset_x(p.x);
        int yoff = get_offset_y(p.y);
        assert(xoff <= dimx);
        assert(yoff <= dimy);
        return get_id(xoff, yoff);
    }
    __host__ __device__ box get_pixel_box(int x, int y){
        const double start_x = mbr->low[0];
        const double start_y = mbr->low[1];

        double lowx = start_x + x * step_x;
        double lowy = start_y + y * step_y;
        double highx = start_x + (x + 1) * step_x;
        double highy = start_y + (y + 1) * step_y;

        return box(lowx, lowy, highx, highy);
    }
    __host__ __device__ int get_numSequences(int id){
        if(pixels->show_status(id) != BORDER) return 0;
        return pixels->pointer[id + 1] - pixels->pointer[id];
    }

    // query
    __host__ __device__ int count_intersection_nodes(Point &p){
        // here we assume the point inside one of the pixel
        int pix_id = get_pixel_id(p);
        assert(pixels->show_status(pix_id) == BORDER);
        int count = 0;
        int x = get_x(pix_id) + 1;
        int i = vertical->offset[x], j;
        if(x < dimx) j = vertical->offset[x + 1];
        else j = vertical->numCrosses;
        while(i < j && vertical->intersection_nodes[i] <= p.y){
            count ++;
            i ++;
        }
        return count;
    }
    __host__ __device__ bool contain(Point &p){
        if(!mbr->contain(p)){
        	return false;
        }

        int target = get_pixel_id(p);
        box bx = get_pixel_box(get_x(target), get_y(target));

        if(pixels->show_status(target) == IN) {
        	return true;
        }
        if(pixels->show_status(target) == OUT){
        	return false;
        }

        // return false;

        bool ret = false;

        for(uint16_t e = 0; e < get_numSequences(target); e ++){    
            auto edges = pixels->edge_sequences[pixels->pointer[target] + e];
            auto pos = edges.pos;
            for(int k = 0; k < edges.len; k ++){
                int i = pos + k;
                int j = i + 1;  //ATTENTION
                if(((vs->p[i].y >= p.y) != (vs->p[j].y >= p.y))){
					double int_x = (vs->p[j].x - vs->p[i].x) * (p.y - vs->p[i].y) / (vs->p[j].y - vs->p[i].y) + vs->p[i].x;
					if(p.x <= int_x && int_x <= bx.high[0]){
						ret = !ret;
					}
				}
            }
        }

        int nc = count_intersection_nodes(p);
        if(nc%2==1){
			ret = !ret;
		}
        return ret;
    }
};