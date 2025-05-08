# CUDA N-Body Simulation

This project simulates gravitational interactions between particles using an brute-force method implemented in CUDA C++. The code runs efficiently on GPU-accelerated systems such as Centaurus, supporting benchmarks of different ranges such as 1000, 10000, and 100000 particles.

---

## Compilation

Compile using `nvcc`:

```bash
nvcc -O3 -arch=sm_61 -o nbody_cuda nbody_cuda.cu
