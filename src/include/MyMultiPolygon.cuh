#pragma once
#include "MyPolygon.cuh"

class MyMultiPolygon{
	std::vector<MyPolygon *> polygons;
public:
	MyMultiPolygon() = default;
	~MyMultiPolygon() = default;
	int num_polygons(){
		return polygons.size();
	}
	void print(){
		std::cout << "MULTIPOLYGON (";
		for(int i = 0; i < polygons.size(); i++){
			if(i > 0){
				std::cout << ",";
			}
			polygons[i]->print_without_head(true);
		}
		std::cout << ")" << std::endl;
	}
	size_t insert_polygon(MyPolygon *p){
		polygons.push_back(p);
		return polygons.size();
	}
};