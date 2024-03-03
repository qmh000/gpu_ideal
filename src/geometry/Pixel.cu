#include "../include/Pixel.cuh"
#include <cstring>

#define BOX_GLOBAL_MIN 100000.0
#define BOX_GLOBAL_MAX -100000.0

__host__ __device__ box::box(){
	low[0] = BOX_GLOBAL_MIN;
	low[1] = BOX_GLOBAL_MIN;
	high[0] = BOX_GLOBAL_MAX;
	high[1] = BOX_GLOBAL_MAX;
}

double box::get_lowx(){return low[0];}
double box::get_lowy(){return low[1];}
double box::get_highx(){return high[0];}
double box::get_highy(){return high[1];}

void box::set_box(double lowx, double lowy, double highx, double highy){
	low[0] = lowx;
	low[1] = lowy;
	high[0] = highx;
	high[1] = highy;
}

Pixel::Pixel(int num_pixels){
	numPixels = num_pixels;
	// status = new uint8_t[num_pixels / 4 + 1];
	status = new int[num_pixels];
    memset(status, 0, num_pixels * sizeof(int));

	pointer = new int[num_pixels + 1];    //这里+1是为了让pointer[num_pixels] = len_edge_sequences，这样对于最后一个pointer就不用特判了

}

__host__ __device__ Pixel::~Pixel(){
    if(status != nullptr) delete []status;
	if(pointer != nullptr) delete []pointer;
	if(edge_sequences != nullptr) delete []edge_sequences;
}

void Pixel::init_edge_sequences(int num_edgeSeqs){
	totalLength = num_edgeSeqs;
	edge_sequences = new EdgeSequence[num_edgeSeqs];
}

int Pixel::get_numPixels(){return numPixels;}
int Pixel::get_totalLength(){return totalLength;}

void Pixel::add_edgeOffset(int id, int off){
	pointer[id] = off;
}

void Pixel::add_edge(int idx, int start, int end){
	edge_sequences[idx].pos = start;
	edge_sequences[idx].len = end - start  + 1;
}

__host__ __device__ void Pixel::set_status(int id, PartitionStatus state){
	status[id] = state;
}

void Pixel::process_null(int x, int y){
	pointer[(x+1)*(y+1)] = totalLength;
	for(int i = (x+1)*(y+1)-1; i >= 0; i --){
		if(show_status(i) != BORDER){
			pointer[i] = pointer[i + 1]; 
		}
	}
}