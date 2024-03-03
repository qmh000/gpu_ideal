#pragma once

class Point{
public:
    double x = 0;
    double y = 0;


public:
    Point() = default;
    ~Point() = default;
    Point(double x_, double y_);

    double get_x();
    double get_y();
};