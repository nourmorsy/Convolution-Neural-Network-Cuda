#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
//#include <omp.h>
extern "C" {
#include "layers.h"
}

#include "volume.h"

//extern double volume_get(volume_t *v, int x, int y, int d);
// Performs the forward pass for a convolutional layer by convolving each one
// of the filters with a particular input, and placing the result in the output
// array.
//
// One way to think about convolution in this case is that we have one of the
// layer's filters (a 3D array) that is superimposed on one of the layer's
// inputs (a second 3D array) that has been implicitly padded with zeros. Since
// convolution is a sum of products (described below), we don't actually have
// to add any zeros to the input volume since those terms will not contribute
// to the convolution. Instead, for each position in the filter, we just make
// sure that we are in bounds for the input volume.
//
// Essentially, the filter is "sliding" across the input, in both the x and y
// directions, where we increment our position in each direction by using the
// stride parameter.
//
// At each position, we compute the sum of the elementwise product of the filter
// and the part of the array it's covering. For instance, let's consider a 2D
// case, where the filter (on the left) is superimposed on some part of the
// input (on the right).
//
//   Filter             Input
//  -1  0  1           1  2  3
//  -1  0  1           4  5  6
//  -1  0  1           7  8  9
//
// Here, the sum of the elementwise product is:
//    Filter[0][0] * Input[0][0] + Filter[0][1] * Input[0][1] + ...
//    = -1 * 1 + 0 * 2 + ... + 0 * 8 + 1 * 9
//    = 6
//
// The 3D case is essentially the same, we just have to sum over the other
// dimension as well. Also, since volumes are internally represented as 1D
// arrays, we must use the volume_get and volume_set commands to access elements
// at a coordinate (x, y, d). Finally, we add the corresponding bias for the
// filter to the sum before putting it into the output volume.
unsigned long int N=4096*4096;


/*
   Here, your GPU kernel, modify the function header as well as kernel launch
   */
   __global__ void doGPU(conv_layer_t *layers,volume_t *in, volume_t *out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	int  x ,y,in_x , in_y;
        double sum = 0.0;
	x= col + layers->pad;
	y =row + layers->pad;
	out_x = blockIdx.x;
	out_y = blockIdx.y;
        volume_t *filter = layers->filters[blockId.x];
        for(int fy = 0; fy < filter->height; fy++){
                int in_y = y + fy;
                for(int fx = 0; fx < filter->width; fx++) {
                        int in_x = x + fx;
                        if(in_y >= 0 && in_y < in->height && in_x >=0 && in_x < in->width){
                                for(int fd = 0; fd < filter->depth; fd++) {
                                        sum += filter->weights[((filter->width * fy) + fx) * filter->depth + fd] * in->weights[((in->width * in_y) + in_x) * in->depth + fd];
                                }
                        }
                }
        }

        sum += layers->biases->weights[blockIdx.x];
        out->weights[((out->width * out_y) + out_x) * out->depth + f_index] = sum; 
extern "C" {


void conv_forward_cu(conv_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
	dim3 dimGrid(16, 1);
	dim3 dimBlock(32,32);
	doGPU<<<gridsize, blocksize>>>(d_conv_layer);
}

}