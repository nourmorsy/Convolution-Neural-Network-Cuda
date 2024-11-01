# CNN with Cuda

## Project Title

### Overview and Goal
This project focuses on [brief description of the project's objective, e.g., developing efficient computational layers for a specific application]. The main goal is to provide high-performance computation by leveraging both CPU and GPU resources.

---

### Dependencies
To run this project, ensure you have the following installed:
- GCC or Clang compiler for C/C++ files
- CUDA toolkit for GPU support

---

### Usage and File Description

#### Files
- **`layers.c`**: Contains core functionalities, primarily for CPU-based computation.
- **`layers_cu.cu`**: Contains GPU-optimized functions implemented using CUDA for efficient processing.

#### Running the Project
To compile and execute the project:

1. Compile the C file:
   ```bash
   gcc layers.c -o layers
   ```
2. Compile the CUDA file:
   ```bash
   nvcc layers_cu.cu -o layers_gpu
   ```
3. Run the files:
   
   - **CPU version:**
     ```bash
     ./layers
     ```
   - **GPU version:**
     ```bash
     ./layers_gpu
     ```
---     
### Acknowledgments
  Special thanks to the teaching assistant for support and guidance:[Nagy K. Aly](https://github.com/nagyaly)



