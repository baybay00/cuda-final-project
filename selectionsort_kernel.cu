#define BLOCK_SIZE 1024

__global__ void selectionsort_kernel(float* in, unsigned size, unsigned pass)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < size - pass)
    {
        int min_idx = pass;
        if(in[tid + pass] < in[min_idx])
        {
            min_idx = tid + pass;
        }

        __syncthreads();

        if (tid == pass && min_idx != pass) {
            int temp = in[pass];
            in[pass] = in[min_idx];
            in[min_idx] = temp;
        }
    }
}