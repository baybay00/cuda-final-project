#define BLOCK_SIZE 1024

__device__ int partition_kernel(float* in, int low, int high)
{
    float pivot = in[high];
    int i = low - 1;
    
    for(int j = low; j <= high - 1; j++)
    {
        if(in[j] <= pivot)
        {
            i++;
            float temp = in[i];
            in[i] = in[j];
            in[j] = temp;
        }
    }

    float temp = in[i + 1];
    in[i + 1] = in[high];
    in[high] = temp;
    
    return i + 1;
}

__global__ void quicksort_kernel(float* in, int low, int high, int* stack)
{
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int start = blockIdx.x*BLOCK_SIZE;
    int end = min(start + BLOCK_SIZE, high);
    

    if(start + tid <= end)
    {
        shared_data[tid] = in[start + tid];
    }    
    __syncthreads();

    int pLow = 0;
    int pHigh = end - start;

    while(pLow < pHigh)
    {
        int pivot_idx = partition_kernel(shared_data, pLow, pHigh);

        if(tid < pivot_idx)
        {
            //left sort of part
        } else if(tid > pivot_idx){
            //right sort of partition
        }

        __syncthreads();
    }
    if(start + tid <= end)
    {
        in[start + tid] = shared_data[tid];
    }
}