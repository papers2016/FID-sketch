#  The LSketch

## Introduction

A *sketch* is a probabilistic data structure that is used to record frequencies of different items in a multi-set.
Various types of sketches have been proposed. The most widely applied among them is the *Count-Min Sketch* for its simplicity, fast processing speed as well as high accuracy. We found that with the assistance of on-chip/off-chip architecture, the accuracy of sketch can be further improved while not losing the fast query speed or structure simplicity. We proposed the new sketch as the *L Sketch*, and gave a high performance prototype using CUDA in this respository.

## How to use

CUDA source codes reside in `main.cu`, `sketch.h` and `hash.h`. To compile, you first need to have ~nvcc~ with the latest version installed on your `Unix` systems. Then type in the following line in shell

    $ nvcc -o sketch main.cu
    
If the compilation succeeds, you can run the executable file like this:

    $ ./sketch 
    
And the results will be written to the `sketch.out` automatically.

## Input & Output format

For input files, each line represents a sequence of chars (ranging from -128 to 127). These strings will feed to the hash functions and determine the relative position in each sketch row. For insert file, another integer is appended to each line, indicating the value to increase the counter by. The output file contains the query results for each query, each integer representing a query result for the query string in corresponding line. All the data are generated randomly and the data size is described in the header file `sketch.h`.
