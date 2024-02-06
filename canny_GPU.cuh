#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

struct kernel_weights{
    char x[7][7];
    char y[7][7];
};

void init_kernel_weights(kernel_weights &, unsigned char);
void cannyGPU(unsigned char* img_h, unsigned char* out, short rows, 
    short cols, unsigned char kernel_size, int low_threshold,
    int high_threshold, unsigned char L2_norm);