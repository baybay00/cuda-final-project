#define BLOCK_SIZE 1024

__global__ void insertionsort_kernel(float* in, unsigned size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int key;

    if(tid < size)
    {
        for(int i = 1; i < size; i++)
        {
            key = in[i];
            int j = i - 1;

            while (j >= 0 && in[j] > key) {
            in[j + 1] = in[j];
            j = j - 1;
        }
        in[j + 1] = key;
        }
    }
}