#include "../include/vertex_sequence.cuh"
#include <cassert>
#include <iostream>

VertexSequence::VertexSequence(int nv){
	p = new Point[nv];
	numVertices = nv;
}

__host__ __device__ VertexSequence::~VertexSequence(){
    if(p != nullptr){
        delete []p;
    }
}

box *VertexSequence::getMBR(){
	box *mbr = new box();
	for(int i = 0; i < numVertices; i ++){
		double lowx = min(mbr->get_lowx(), p[i].get_x());
		double highx = max(mbr->get_highx(), p[i].get_x());
		double lowy = min(mbr->get_lowy(), p[i].get_y());
		double highy = max(mbr->get_highy(), p[i].get_y());
		mbr->set_box(lowx, lowy, highx, highy);
	}
	return mbr;
}

int VertexSequence::get_numVertices(){return numVertices;}
double VertexSequence::get_pointX(int idx){return p[idx].get_x();}
double VertexSequence::get_pointY(int idx){return p[idx].get_y();}

void VertexSequence::print(bool complete_ring){
	std::cout << "(";
	for(int i=0;i<numVertices;i++){
		if(i!=0){
			std::cout << ",";
		}
		printf("%f ",p[i].x);
		printf("%f",p[i].y);
	}
	// the last vertex should be the same as the first one for a complete ring
	if(complete_ring){
		if(p[0].x!=p[numVertices-1].x||p[0].y!=p[numVertices-1].y){
			std::cout << ",";
			printf("%f ",p[0].x);
			printf("%f",p[0].y);
		}
	}
	std::cout << ")";
}

size_t VertexSequence::decode(char *source){
	size_t decoded = 0;
	numVertices = ((size_t *)source)[0];
	assert(numVertices>0);
	p = new Point[numVertices + 1];
	decoded += sizeof(size_t);
	memcpy((char *)p,source+decoded,numVertices*sizeof(Point));
	decoded += numVertices*sizeof(Point);
	p[numVertices ++] = p[0];
	return decoded;
}

