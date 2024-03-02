#include "../include/MyRaster.cuh"
#include <cassert>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

MyRaster::~MyRaster(){
	if(mbr != nullptr) delete mbr;
	if(pixels != nullptr) delete pixels;
	if(horizontal != nullptr) delete horizontal;
	if(vertical != nullptr) delete vertical;
}

MyRaster::MyRaster(VertexSequence *vst, int epp){
	assert(epp>0);
	vs = vst;
	mbr = vs->getMBR();

	double multi = abs((mbr->get_highy()-mbr->get_lowy())/(mbr->get_highx()-mbr->get_lowx()));
	dimx = std::pow((vs->get_numVertices()/epp)/multi, 0.5);
	dimy = dimx * multi;

	if(dimx==0) dimx = 1;
	if(dimy==0) dimy = 1;

	step_x = (mbr->get_highx()-mbr->get_lowx())/dimx;
	step_y = (mbr->get_highy()-mbr->get_lowy())/dimy;

	if(step_x < 0.00001){
		step_x = 0.00001;
		dimx = (mbr->get_highx()-mbr->get_lowx())/step_x + 1;
	}
	if(step_y < 0.00001){
		step_y = 0.00001;
		dimy = (mbr->get_highy()-mbr->get_lowy())/step_y + 1;
	}
}

int MyRaster::get_id(int x, int y){
	assert(x>=0&&x<=dimx);
	assert(y>=0&&y<=dimy);
	return y * (dimx+1) + x;
}

// from id to pixel x
int MyRaster::get_x(int id){
	return id % (dimx+1);
}

// from id to pixel y
int MyRaster::get_y(int id){
	assert((id / (dimx+1)) <= dimy);
	return id / (dimx+1);
} 

void MyRaster::process_crosses(std::map<int, std::vector<cross_info>> edges_info){
	int num_edge_seqs = 0;
	for(auto ei : edges_info){
		num_edge_seqs += ei.second.size();
	}
	pixels->init_edge_sequences(num_edge_seqs);

	int idx = 0;
	for(auto info : edges_info){
		auto pix = info.first;
		auto crosses = info.second; 
		if(crosses.size() == 0) return;
		
		if(crosses.size() % 2 == 1){
			crosses.push_back(cross_info((cross_type)!crosses[crosses.size()-1].type, crosses[crosses.size()-1].edge_id));
		}
		
		assert(crosses.size()%2==0);

		// 根据crosses.size()，初始化
		int start = 0;
		int end = crosses.size() - 1;
		pixels->add_edgeOffset(pix, idx);

		if(crosses[0].type == LEAVE){
			assert(crosses[end].type == ENTER);
			pixels->add_edge(idx ++, 0, crosses[0].edge_id);
			pixels->add_edge(idx ++, crosses[end].edge_id, vs->get_numVertices() - 2);
			start ++;
			end --;
		}

		for(int i = start; i <= end; i++){
			assert(crosses[i].type == ENTER);
			//special case, an ENTER has no pair LEAVE,
			//happens when one edge crosses the pair
			if(i == end || crosses[i + 1].type == ENTER){
				pixels->add_edge(idx ++, crosses[i].edge_id, crosses[i].edge_id);
			}else{
				pixels->add_edge(idx ++, crosses[i].edge_id, crosses[i+1].edge_id);
				i++;
			}
		}
	}
}

void MyRaster::process_intersection(std::map<int, std::vector<double>> intersection_info, std::string direction){
	int num_nodes = 0;
	for(auto i : intersection_info){
		num_nodes += i.second.size();
	}
	if(direction == "horizontal"){
		horizontal->init_intersection_nodes(num_nodes);
		int idx = 0;
		for(auto info : intersection_info){
			auto h = info.first;
			auto nodes = info.second;
			
			sort(nodes.begin(), nodes.end());

			horizontal->set_offset(h, idx);

			for(auto node : nodes){
				horizontal->add_node(idx, node);
				idx ++;
			}
		}
		horizontal->set_offset(dimy, idx);
	}else{
		vertical->init_intersection_nodes(num_nodes);
		int idx = 0;
		for(auto info : intersection_info){
			auto h = info.first;
			auto nodes = info.second;
			
			sort(nodes.begin(), nodes.end());

			vertical->set_offset(h, idx);

			for(auto node : nodes){
				vertical->add_node(idx, node);
				idx ++;
			}
		}
		vertical->set_offset(dimx, idx);		
	}
}

void MyRaster::process_pixels(int x, int y){
	pixels->process_null(x, y);
}

void MyRaster::init_pixels(){
	assert(mbr);
	pixels = new Pixel((dimx+1)*(dimy+1));
	horizontal =  new Grid_line(dimy + 1);
	vertical = new Grid_line(dimx + 1);
}

void MyRaster::evaluate_edges(){
	std::map<int, std::vector<double>> horizontal_intersect_info;
	std::map<int, std::vector<double>> vertical_intersect_info;
	std::map<int, std::vector<cross_info>> edges_info;

	// normalize
	assert(mbr);
	const double start_x = mbr->get_lowx();
	const double start_y = mbr->get_lowy();

	for(int i=0; i < vs->get_numVertices() - 1; i++){
		double x1 = vs->get_pointX(i);
		double y1 = vs->get_pointY(i);
		double x2 = vs->get_pointX(i + 1);
		double y2 = vs->get_pointY(i + 1);

		int cur_startx = (x1-start_x)/step_x;
		int cur_endx = (x2-start_x)/step_x;
		int cur_starty = (y1-start_y)/step_y;
		int cur_endy = (y2-start_y)/step_y;

		if(cur_startx==dimx+1){
			cur_startx--;
		}
		if(cur_endx==dimx+1){
			cur_endx--;
		}

		int minx = min(cur_startx,cur_endx);
		int maxx = max(cur_startx,cur_endx);

		if(cur_starty==dimy+1){
			cur_starty--;
		}
		if(cur_endy==dimy+1){
			cur_endy--;
		}
		// todo should not happen for normal cases
		if(cur_startx>dimx||cur_endx>dimx||cur_starty>dimy||cur_endy>dimy){
			
			std::cout << "xrange\t" << cur_startx << " " << cur_endx << std::endl;
			std::cout << "yrange\t" << cur_starty << " " << cur_endy << std::endl;
			printf("xrange_val\t%f %f\n",x1, x2);
			printf("yrange_val\t%f %f\n",y1, y2);
			std::cout << "dim\t" << dimx << " " << dimy << std::endl;
			std::cout << "box\t" << mbr->get_lowx() << " " << mbr->get_lowy() << " " << mbr->get_highx() << " " << mbr->get_highy() << std::endl;
			assert(false);
		}
		assert(cur_startx<=dimx);
		assert(cur_endx<=dimx);
		assert(cur_starty<=dimy);
		assert(cur_endy<=dimy);

		//in the same pixel
		if(cur_startx==cur_endx&&cur_starty==cur_endy){
			continue;
		}

		if(y1==y2){
			//left to right
			if(cur_startx<cur_endx){
				for(int x=cur_startx;x<cur_endx;x++){
					vertical_intersect_info[x + 1].push_back(y1);
					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(x + 1, cur_starty)].push_back(cross_info(ENTER, i));
					pixels->set_status(get_id(x, cur_starty), BORDER);
					pixels->set_status(get_id(x+1, cur_starty), BORDER);
				}
			}else { // right to left
				for(int x=cur_startx;x>cur_endx;x--){
					vertical_intersect_info[x].push_back(y1);
					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(x - 1, cur_starty)].push_back(cross_info(ENTER, i));
					pixels->set_status(get_id(x, cur_starty), BORDER);
					pixels->set_status(get_id(x-1, cur_starty), BORDER);
				}
			}
		}else if(x1==x2){
			//bottom up
			if(cur_starty<cur_endy){
				for(int y=cur_starty;y<cur_endy;y++){

					horizontal_intersect_info[y + 1].push_back(x1);
					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(cur_startx, y + 1)].push_back(cross_info(ENTER, i));
					pixels->set_status(get_id(cur_startx, y), BORDER);
					pixels->set_status(get_id(cur_startx, y+1), BORDER);
				}
			}else { //border[bottom] down
				for(int y=cur_starty;y>cur_endy;y--){
					if(cur_startx>dimx||cur_endx>dimx||cur_startx<0||cur_endx<0){
						std::cout << "xrange\t" << cur_startx << " " << cur_endx << std::endl;
						std::cout << "yrange\t" << cur_starty << " " << cur_endy << std::endl;
						printf("xrange_val\t%lf %lf\n",x1, x2);
						printf("yrange_val\t%lf %lf\n",y1, y2);
						std::cout << "dim\t" << dimx << " " << dimy << std::endl;
						std::cout << "box\t" << mbr->get_lowx() << " " << mbr->get_lowy() << " " << mbr->get_highx() << " " << mbr->get_highy() << std::endl;
						assert(false);
					}
					horizontal_intersect_info[y].push_back(x1);
					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(cur_startx, y - 1)].push_back(cross_info(ENTER, i));
					pixels->set_status(get_id(cur_startx, y), BORDER);
					pixels->set_status(get_id(cur_startx, y-1), BORDER);
				}
			}
		}else{
			// solve the line function
			double a = (y1-y2)/(x1-x2);
			double b = (x1*y2-x2*y1)/(x1-x2);

			int x = cur_startx;
			int y = cur_starty;
			while(x!=cur_endx||y!=cur_endy){
				bool passed = false;
				double yval = 0;
				double xval = 0;
				int cur_x = 0;
				int cur_y = 0;
				//check horizontally
				if(x!=cur_endx){
					if(cur_startx<cur_endx){
						xval = ((double)x+1)*step_x+start_x;
					}else{
						xval = (double)x*step_x+start_x;
					}
					yval = xval*a+b;
					cur_y = (yval-start_y)/step_y;
					
					if(cur_y>max(cur_endy, cur_starty)){
						cur_y=max(cur_endy, cur_starty);
					}
					if(cur_y<min(cur_endy, cur_starty)){
						cur_y=min(cur_endy, cur_starty);
					}
					if(cur_y==y){
						passed = true;
						// left to right
						if(cur_startx<cur_endx){
							vertical_intersect_info[x + 1].push_back(yval);
							pixels->set_status(get_id(x, y), BORDER);
							edges_info[get_id(x ++, y)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							pixels->set_status(get_id(x, y), BORDER);
						}else{//right to left
							vertical_intersect_info[x].push_back(yval);
							pixels->set_status(get_id(x, y), BORDER);
							edges_info[get_id(x --, y)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							pixels->set_status(get_id(x, y), BORDER);
						}
					}
				}
				//check vertically
				if(y!=cur_endy){
					if(cur_starty<cur_endy){
						yval = (y+1)*step_y+start_y;
					}else{
						yval = y*step_y+start_y;
					}
					xval = (yval-b)/a;
					cur_x = (xval-start_x)/step_x;
					//printf("x %f %d\n",(xval-start_x)/step_x,cur_x);
					if(cur_x>max(cur_endx, cur_startx)){
						cur_x=max(cur_endx, cur_startx);
					}
					if(cur_x<min(cur_endx, cur_startx)){
						cur_x=min(cur_endx, cur_startx);
					}
					if(cur_x==x){
						passed = true;
						if(cur_starty<cur_endy){// bottom up
							horizontal_intersect_info[y + 1].push_back(xval);
							pixels->set_status(get_id(x, y), BORDER);
							edges_info[get_id(x, y ++)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							pixels->set_status(get_id(x, y), BORDER);				
						}else{// top down
							horizontal_intersect_info[y].push_back(xval);

							pixels->set_status(get_id(x, y), BORDER);
							edges_info[get_id(x, y --)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							pixels->set_status(get_id(x, y), BORDER);
						}
					}
				}
				assert(passed);
			}
		}
	}

	// 初始化edge_sequences和intersection nodes list;
	process_crosses(edges_info);
	process_intersection(horizontal_intersect_info, "horizontal");
	process_intersection(vertical_intersect_info, "vertical");
	process_pixels(dimx, dimy);
}

void MyRaster::scanline_reandering(){
	const double start_x = mbr->get_lowx();
	const double start_y = mbr->get_lowy();

	for(int y = 1; y < dimy; y ++){
		bool isin = false;
		uint16_t i = horizontal->get_offset(y), j = horizontal->get_offset(y + 1);
		for(int x = 0; x < dimx; x ++){
			if(pixels->show_status(get_id(x, y)) != BORDER){
				if(isin){
					pixels->set_status(get_id(x, y), IN);
				}else{
					pixels->set_status(get_id(x, y), OUT);
				}
				continue;
			}
			int pass = 0;
			while(i < j && horizontal->get_node(i) <= start_x + step_x * (x + 1)){
				pass ++;
				i ++;
			}
			if(pass % 2 == 1) isin = !isin;

		}
	}
}

void MyRaster::rasterization(){
	//1. create space for the pixels
	init_pixels();

	//2. edge crossing to identify BORDER pixels
	evaluate_edges();

	//3. determine the status of rest pixels with scanline rendering
	scanline_reandering();
}

box* MyRaster::get_mbr(){return mbr;}
void MyRaster::set_mbr(box* addr){mbr = addr;}
VertexSequence* MyRaster::get_vs(){return vs;}
void MyRaster::set_vs(VertexSequence* addr){vs = addr;}
Pixel* MyRaster::get_pix(){return pixels;}
void MyRaster::set_pix(Pixel* addr){pixels = addr;}
Grid_line* MyRaster::get_horizontal(){return horizontal;}
void MyRaster::set_horizontal(Grid_line* addr){horizontal = addr;}
Grid_line* MyRaster::get_vertical(){return vertical;}
void MyRaster::set_vertical(Grid_line* addr){vertical = addr;}



