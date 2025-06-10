# Max-Cut Problem

This project implements a solution to the Max-Cut problem using C++.

## Compilation

To compile the program, use the following command:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 main.cpp -o maxcut -lstdc++fs
```

## Execution

After compiling, you can run the program with:

```bash
./maxcut
```

## Requirements

- A C++17 compatible compiler (e.g., g++).
- OpenMP support for parallel processing.

## Description

The program solves the Max-Cut problem, which involves partitioning the vertices of a graph into two subsets such that the number of edges between the subsets is maximized.

## File Structure

- `main.cpp`: Contains the implementation of the Max-Cut algorithm.
- `readme.md`: Documentation for the project.

## Notes

- Ensure that your system supports the `-march=native` flag for optimal performance.
- The `-fopenmp` flag enables multi-threading using OpenMP.
