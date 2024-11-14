#define BLOCK_SIZE 1024

__global__ void bubbleSort(float* in, unsigned size, bool is_even)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = is_even ? 2 * tx : 2 * tx + 1;

    if(i < size - 1 && in[i] > in[i + 1])
    {
        float temp = in[i];
        in[i] = in[i + 1];
        in[i + 1] = temp;
    }
}