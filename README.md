# Particle-Parallelization-University-Project

### Assignment for Parallel Computing with Graphical Processing Units (GPUs) module at the University of Sheffield. Achieved Grade: 70% (First Class).


## About

The assignment required parallelizing a serial particle generation program implemented in `C`. The program consists of five stages:

- `init`: Allocates host and device memory;
- `stage1`: Calculating particle contribution to pixels of the image;
- `stage2`: Calculating pixel data by scanning particle contributions, writing colours to storage and sorting them according to specification;
- `stage3`: Blending the colours one-by-one according to previous calculations to form the final image;
- `free`: Frees host and device memory.

The main objective of the assignment was to improve program performance via the usage of parallelization techniques.

## Development

Each stage was implemented using `OpenMP` and `CUDA` and can be examined in [openmp.c](src/openmp.c) and [cuda.cu](src/cuda.cu) respectively. The main program ([main.cu](src/main.cu)) runs each stage in the provided sequence, whilst measuring performance.

Different program arguments are configured to benchmark the performance via multiple runs to minimise the margin of error.
