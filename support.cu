#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(float **vec_h, unsigned size)
{
    *vec_h = (float*)malloc(size*sizeof(float));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%100)/100.00;
    }

}

void swap(float* xp, float* yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void bubbleSort(float in[], int size)
{
    bool swapped;
    for(int i = 0; i < size - 1; i++)
    {
        swapped = false;
        for(int j = 0; j < size - i; j++)
        {
            if(in[j] > in[j + 1])
            {
                swap(&in[j], &in[j + 1]);
                swapped = true;
            }
        }
        if(swapped == false) break;
    }
}

int partition(float in[], int low, int high)
{
    float pivot = in[high];
    int i = low - 1;
    for(int j = low; j <= high - 1; j++)
    {
        if(in[j] < pivot)
        {
            i++;
            swap(&in[i], &in[j]);
        }
    }

    swap(&in[i + 1], &in[high]);
    return i + 1;
}

void quickSort(float in[], int low, int high)
{
    if(low < high)
    {
        int pi = partition(in, low, high);
        quickSort(in, low, pi - 1);
        quickSort(in, pi + 1, high);
    }
}

void insertionSort(float in[], int size)
{
    for (int i = 1; i < size; ++i) {
        int key = in[i];
        int j = i - 1;

        while (j >= 0 && in[j] > key) {
            in[j + 1] = in[j];
            j = j - 1;
        }
        in[j + 1] = key;
    }
}

void selectionSort(int in[], int size) {
    for (int i = 0; i < size - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < size; j++) {
            if (in[j] < in[min_idx]) {
                min_idx = j;
            }
        }
        int temp = in[i];
        in[i] = in[min_idx];
        in[min_idx] = temp;
    }
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

