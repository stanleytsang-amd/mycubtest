#!/bin/bash
#nvcc -D CUB_SORT_TYPE=uint32_t -lcurand  -o sort_32 device_radix_sort.cu
if [ -z "$1" ]
then 
    echo "Please specify the 'rocm' or 'cuda'."
    exit
fi

if [ "$1" == "cuda" ]
then
    nvcc -D CUB_SORT_TYPE=float -lcurand  -o sort_float_cuda.exe device_radix_sort.cu
else
    hipcc device_radix_sort.cpp -I/opt/rocm/include/hiprand/ -I/opt/rocm/include/rocrand/ -L/opt/rocm/lib/ -lhiprand -D CUB_SORT_TYPE=float -o sort_float_rocm.exe
fi

