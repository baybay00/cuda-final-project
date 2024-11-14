#define BLOCK_SIZE 1024

__global__ void selectionsort_kernel(float *in, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (int i = tid; i < size - 1; i += gridDim.x * blockDim.x) {
        int minIndex = i;

        for (int j = i + 1; j < size; j++) {
            if (in[j] < in[minIndex]) {
                minIndex = j;
            }
        }

        if (minIndex != i) {
            int temp = in[i];
            in[i] = in[minIndex];
            in[minIndex] = temp;
        }
    }
}