#pragma once
#include "Point.h"
#include "Pixel.cuh"

class VertexSequence{
public:
    int numVertices = 0;
    Point *p = nullptr;

public:
    VertexSequence() = default;
    VertexSequence(int nv);
    ~VertexSequence();

    int get_numVertices();
    double get_pointX(int idx);
    double get_pointY(int idx);
    
    box* getMBR();
    
    size_t decode(char *source);
};