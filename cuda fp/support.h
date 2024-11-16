#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void initVector(float **vec_h, unsigned size);
void swap(float* xp, float* yp);
void bubbleSort(float arr[], int n);
void quickSort(float arr[], int low, int high);
void partition(int arr[], int low, int high);
void insertionSort(float arr[], int size);
void selectionSort(float arr[], int size);
void shellSort(float arr[], int size);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...)                                                       \
    do {                                                                      \
        printf("[%s:%d\n%s]", __FILE__, __LINE__, msg);                       \
        exit(-1);                                                             \
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
