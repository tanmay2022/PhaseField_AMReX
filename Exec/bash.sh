#!/bin/bash

make

g++ -o Replace Replace.cpp

./Replace Input.in Filling.in

mpirun -np 64 ./main2d.gnu.MPI.ex input2.in
