#define BLOCK_SIZE 1024

__device__ int partition_kernel(float *in, int left, int right) {
    float pivot = in[right];
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (in[j] <= pivot) {
            i++;
            float temp = in[i];
            in[i] = in[j];
            in[j] = temp;
        }
    }
    float temp = in[i + 1];
    in[i + 1] = in[right];
    in[right] = temp;

    return i + 1;
}

__global__ void quicksort_kernel(float* in, int* stack, int* top, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Stack initialization, done by thread 0
    if (tid == 0) {
        stack[0] = 0;
        stack[1] = size - 1;
        *top = 1;
    }

    __syncthreads();  // Ensure stack initialization is done before proceeding

    while (*top >= 0) {
        // Pop the top element from the stack
        int right = stack[*top];
        int left = stack[*top - 1];
        *top -= 2;

        // Perform partitioning
        int pivotIdx = partition_kernel(in, left, right);

        // Push subarrays to the stack
        if (pivotIdx - 1 > left) {
            stack[*top + 1] = left;
            stack[*top + 2] = pivotIdx - 1;
            *top += 2;
        }
        if (pivotIdx + 1 < right) {
            stack[*top + 1] = pivotIdx + 1;
            stack[*top + 2] = right;
            *top += 2;
        }
    }
}