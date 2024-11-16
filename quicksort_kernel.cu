#define BLOCK_SIZE 256
#define STACK_SIZE 1024

__device__ int partition_d(float *arr, int low, int high) {
    float pivot = arr[high]; 
    int i = low - 1; 

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
        
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    
    float temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}

__global__ void quicksort_kernel(float *arr, int low, int high) {

    __shared__ int stack[STACK_SIZE];  
    int stackIdx = threadIdx.x;
    int top = -1;

    if (stackIdx == 0) {
        stack[++top] = low;
        stack[++top] = high;
    }

    __syncthreads();

    while (top >= 0) {
        int h = stack[top--];  
        int l = stack[top--]; 

        int p = partition_d(arr, l, h);

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

