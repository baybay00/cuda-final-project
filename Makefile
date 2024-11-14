
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64

default: bubble_sort quick_sort insertion_sort selection_sort shell_sort

bubble_main.o: bubblesort_main.cu bubblesort_kernel.cu support.h
	$(NVCC) -c -o $@ bubblesort_main.cu $(NVCC_FLAGS)

quick_main.o: quicksort_main.cu quicksort_kernel.cu support.h
	$(NVCC) -c -o $@ quicksort_main.cu $(NVCC_FLAGS)

insertion_main.o: insertionsort_main.cu insertionsort_kernel.cu support.h
	$(NVCC) -c -o $@ insertionsort_main.cu $(NVCC_FLAGS)

selection_main.o: selectionsort_main.cu selectionsort_kernel.cu support.h
	$(NVCC) -c -o $@ selectionsort_main.cu $(NVCC_FLAGS)

shell_main.o: shellsort_main.cu shellsort_kernel.cu support.h
	$(NVCC) -c -o $@ shellsort_main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

bubble_sort: bubble_main.o support.o
	$(NVCC) bubble_main.o support.o -o bubble_sort $(LD_FLAGS)

quick_sort: quick_main.o support.o
	$(NVCC) quick_main.o support.o -o quick_sort $(LD_FLAGS)

insertion_sort: insertion_main.o support.o
	$(NVCC) insertion_main.o support.o -o insertion_sort $(LD_FLAGS)

selection_sort: selection_main.o support.o 
	$(NVCC) selection_main.o support.o -o selection_sort $(LD_FLAGS)

shell_sort: shell_main.o support.o 
	$(NVCC) shell_main.o support.o -o shell_sort $(LD_FLAGS)

clean:
	rm -rf *.o bubble_sort quick_sort insertion_sort selection_sort shell_sort
