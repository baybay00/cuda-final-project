#define BLOCK_SIZE 256
#define STACK_SIZE 1024  // Adjust size according to your GPU

__device__ int partition_d(float *arr, int low, int high) {
    float pivot = arr[high];  // Pivot element
    int i = low - 1;  // Index of smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            // Swap arr[i] and arr[j]
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Swap arr[i + 1] and arr[high] (or pivot)
    float temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}

__global__ void quicksort_kernel(float *arr, int low, int high) {
    // Shared memory stack for subarray bounds (low, high)
    __shared__ int stack[STACK_SIZE];  
    int stackIdx = threadIdx.x;
    int top = -1;

    // Start with the full array range
    if (stackIdx == 0) {
        stack[++top] = low;
        stack[++top] = high;
    }

    __syncthreads();

    // Loop until the stack is empty
    while (top >= 0) {
        int h = stack[top--];  // Pop high
        int l = stack[top--];  // Pop low

        // Partition the array
        int p = partition_d(arr, l, h);

        // Push the subarrays onto the stack
        if (p - 1 > l) {
            stack[++top] = l;
            stack[++top] = p - 1;
        }
        if (p + 1 < h) {
            stack[++top] = p + 1;
            stack[++top] = h;
        }
    }
}

