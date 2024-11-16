#include <stdio.h>
#include "support.h"
#include "quicksort_kernel.cu"

int main(int argc, char* argv[]) {
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
        in_elements = 100000;
    } else if(argc == 2) {
        in_elements = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./quicksort         # Input of size 100,000 is used"
           "\n    Usage: ./quicksort <m>      # Input of size m is used"
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
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory: in");

    int *d_stack, *d_top;
    cuda_ret = cudaMalloc((void**)&d_stack, 2 * in_elements * sizeof(int));  // Stack size can be up to 2*in_elements
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for stack");

    cuda_ret = cudaMalloc((void**)&d_top, sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for top");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, in_elements * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device: in");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Timing host-side bubble sort ---------------------------------------
    printf("Running host-side quick sort...\n"); fflush(stdout);
    startTime(&timer);

    quickSort(in_h, 0, in_elements - 1);

    stopTime(&timer); 
    printf("Host-side quick sort time: %f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel...\n"); fflush(stdout);
    startTime(&timer);

    dim_block.x = BLOCK_SIZE;
    dim_grid.x = (in_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Adjust grid size

    quicksort_kernel<<<dim_grid, dim_block>>>(in_d, d_stack, d_top, in_elements);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    
    stopTime(&timer); 
    printf("Device quick sort time: %f s\n", elapsedTime(timer));

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
    cudaFree(d_stack);
    cudaFree(d_top);
    free(in_h);
    free(sorted_h);

    return 0;
}