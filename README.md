#  The FID-sketch

## Introduction

A *sketch* is a probabilistic data structure that is used to record frequencies of different items in a multi-set.
Various types of sketches have been proposed. The most widely applied among them is the *Count-Min sketch* for its simplicity, fast processing speed as well as high accuracy. We found that with the assistance of on-chip/off-chip architecture, the accuracy of sketch can be further improved while not losing the fast query speed or structure simplicity. We proposed the new sketch as the *FID-sketch*,
and gave a high performance implementation using CUDA as well as other relative sketches in C++ for reference.

## How to use

Suppose you've already cloned the respository and start from the `FID-sketch` directory.

###### For CUDA implementation:

First enter `CUDA` directory:

    $ cd CUDA

CUDA source codes reside in `main.cu`, `sketch.h` and `hash.h`. To compile, you first need to have `nvcc` with the latest version installed on your `Unix` systems. Then type in the following line in shell

    $ nvcc -o sketch main.cu
    
If the compilation succeeds, you can run the executable file like this:

    $ ./sketch 
    
And the results will be written to the `sketch.out` automatically.

###### For C++ implementation:

This part of source code contains implementation of the *CM-sketch*, *Count-sketch*, *CU-sketch*, *CMl-sketch*, and *FID-sketch* written in C++. First you need to enter `CPU` directory:

    $ cd CPU
    
The `data.cpp` is for test data sets generation. Compile and run the generated executable like this:

    $ g++ -o data data.cpp
    $ ./data
    
You will get the test input file `insert.dat` and `query.dat`. The source code mainly resides in `sketch.h` and the main routine is in `main.cpp`. You can compile the sketch source code like this:

    $ g++ -std=c++11 -o sketch main.cu
    
and run the executable `sketch` to perform experiments on the test input files `insert.dat` and `query.dat`.

## Input & Output format

For input files, each line represents a sequence of chars (ranging from -128 to 127). These strings will feed to the hash functions and determine the relative position in each sketch row. For insert file, another integer is appended to each line, indicating the value to increase the counter by. The output file contains the query results for each query, each integer representing a query result for the query string in corresponding line. All the data are generated randomly and the data size is described in the header file `sketch.h`. Note that for the main routine in `CPU/main.cpp`, it does not produce output for query results now, but you may add the query tests easily if you wish.
