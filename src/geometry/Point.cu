#include "../include/Point.cuh"

Point::Point(double _x, double _y){
    x = _x;
    y = _y;
}

double Point::get_x(){return x;}
double Point::get_y(){return y;}