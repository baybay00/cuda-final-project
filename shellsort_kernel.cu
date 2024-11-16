#define BLOCK_SIZE 128

__global__ void shellsort_kernel(float* in, unsigned size, unsigned gap)
{
    extern __shared__ float sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        sdata[threadIdx.x] = in[tid];
        __syncthreads();
        for (int i = tid; i < size - 1; i += blockDim.x * gridDim.x)
        {
            for (int j = i + gap; j < size; j += gap)
            {
                float temp = sdata[j];
                int k = j;

                while (k >= gap && sdata[k - gap] > temp)
                {
                    sdata[k] = sdata[k - gap];
                    k -= gap;
                }
                sdata[k] = temp;
            }
        }
        __syncthreads();
        if (tid < size) {
            in[tid] = sdata[threadIdx.x];
        }
    }
}
