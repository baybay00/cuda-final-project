#define BLOCK_SIZE 1024

__global__ void shellsort_kernel(float* in, unsigned size, unsigned gap)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = tid; i < size - 1; i += blockDim.x * gridDim.x)
    {
        for(int j = i + gap; j < size; j += gap)
        {
            float temp = in[j];
            int k = j;

            while(k >= gap && in[k - gap] > temp)
            {
                in[k] = in[k - gap];
                k -= gap;
            }
            in[k] = temp;
        }
    }
}