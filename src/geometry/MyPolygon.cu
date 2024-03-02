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