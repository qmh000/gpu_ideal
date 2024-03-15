#!/bin/bash

make clean
make -j16 $1
../build/$1
