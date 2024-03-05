#include "../include/MyPolygon.cuh"

MyPolygon::MyPolygon(){
	pthread_mutex_init(&ideal_partition_lock, NULL);
}

MyPolygon::~MyPolygon(){
	if(mbr != nullptr) delete mbr;
	if(raster != nullptr) delete raster;
	if(boundary != nullptr) delete boundary;
}

int MyPolygon::get_numVertices(){
    if(!boundary) return 0;
    return boundary->get_numVertices();
}

void MyPolygon::print_without_head(bool complete_ring){
	assert(boundary);
	std::cout << "(";
	boundary->print(complete_ring);
	std::cout << ")";
}

void MyPolygon::print(bool complete_ring){
	std::cout << "POLYGON";
	print_without_head(complete_ring);
	std::cout << std::endl;
}

box* MyPolygon::getMBR(){
    if(mbr) return mbr;
	mbr = boundary->getMBR();
	return mbr;

}

void MyPolygon::rasterization(int vpr){
	assert(vpr>0);
	if(raster) return;
	pthread_mutex_lock(&ideal_partition_lock);
	raster = new MyRaster(boundary, vpr);
	raster->rasterization();
	pthread_mutex_unlock(&ideal_partition_lock);
}

size_t MyPolygon::decode(char *source){
	size_t decoded = 0;
	assert(!boundary);
	boundary = new VertexSequence();
	size_t num_holes = ((size_t *)source)[0];
	decoded += sizeof(size_t);
	decoded += boundary->decode(source+decoded);
	for(size_t i=0;i<num_holes;i++){
		VertexSequence *vs = new VertexSequence();
		decoded += vs->decode(source+decoded);
	}
	return decoded;
}

MyPolygon *MyPolygon::gen_box(double min_x,double min_y,double max_x,double max_y){
	MyPolygon *mbr = new MyPolygon();
	mbr->boundary = new VertexSequence(5);
	mbr->boundary->p[0].x = min_x;
	mbr->boundary->p[0].y = min_y;
	mbr->boundary->p[1].x = max_x;
	mbr->boundary->p[1].y = min_y;
	mbr->boundary->p[2].x = max_x;
	mbr->boundary->p[2].y = max_y;
	mbr->boundary->p[3].x = min_x;
	mbr->boundary->p[3].y = max_y;
	mbr->boundary->p[4].x = min_x;
	mbr->boundary->p[4].y = min_y;
	return mbr;
}

MyPolygon *MyPolygon::gen_box(box &pix){
	return gen_box(pix.low[0],pix.low[1],pix.high[0],pix.high[1]);
}