#!/bin/bash
mpicxx -O3 -DCUDA_ENABLED -std=c++11 -c utils.cpp num_method.cpp main.cpp
nvcc -O3 -c cuda.cu
mpicxx -O3 -o Task4_Cuda -L/usr/local/cuda/lib64 -lcuda -lcudart utils.o cuda.o num_method.o main.o