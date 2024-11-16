#include <stdio.h>

#include "support.h"
#include "shellsort_kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *in_h;
    float *in_d;
    unsigned in_elements;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;

    // Allocate and initialize host memory
    if(argc == 1) {
        in_elements = 10000;
    } else if(argc == 2) {
        in_elements = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./shell_sort          # Input of size 10,000 is used"
           "\n    Usage: ./shell_sort <m>      # Input of size m is used"
           "\n");
        exit(0);
    }
    initVector(&in_h, in_elements);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", in_elements);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&in_d, in_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, in_elements * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Timing host-side shell sort ---------------------------------------
    printf("Running host-side shell sort...\n"); fflush(stdout);
    startTime(&timer);

    float* in_h_copy = (float*)malloc(in_elements * sizeof(float));
    memcpy(in_h_copy, in_h, in_elements * sizeof(float));
    shellSort(in_h_copy, in_elements); 

    stopTime(&timer); 
    printf("Host-side shell sort time: %f s\n", elapsedTime(timer));

    free(in_h_copy);  // Free the temporary copy

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel...\n"); fflush(stdout);
    startTime(&timer);

    dim_block.x = BLOCK_SIZE;
    dim_grid.x = (in_elements + BLOCK_SIZE -1)/(BLOCK_SIZE);

    for(int gap = in_elements/2; gap > 0; gap /= 2)
    {
        shellsort_kernel<<<dim_grid, dim_block, dim_block.x*sizeof(float)>>>(in_d, in_elements, gap);
    }
    // cuda_ret = cudaDeviceSynchronize();
    // if(cuda_ret != cudaSuccess) printf("unable to launch kernel");
    
    stopTime(&timer); 
    printf("Device shell sort time: %f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...\n"); fflush(stdout);

    float* sorted_h = (float*)malloc((in_elements) * sizeof(float));
    cudaMemcpy(sorted_h, in_d, in_elements*sizeof(float), cudaMemcpyDeviceToHost);

    bool sorted = true;
    for (int i = 1; i < in_elements; i++) {
        if (sorted_h[i] < sorted_h[i - 1]) {
            sorted = false;
            break;
        }
    }

    if(sorted)
    {
        printf("Sorting successful\n\n");
    } else {
        printf("Sorting failed\n\n");
    }
    // Free memory ------------------------------------------------------------

    cudaFree(in_d);
    free(in_h);
    free(sorted_h);

    return 0;
}

