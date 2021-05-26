#!/bin/bash
#nvcc -D CUB_SORT_TYPE=uint32_t -lcurand  -o sort_32 device_radix_sort.cu
if [ -z "$1" ]
then 
    echo "Please specify the 'rocm' or 'cuda'."
    exit
fi

if [ "$1" == "cuda" ]
then
    nvcc --std=c++11 -DCUB_SORT_TYPE=float -lcurand  -o sort_float_cuda.exe device_radix_sort.cu
else
    hipcc device_radix_sort.cpp -I/opt/rocm/hiprand/include -I/opt/rocm/rocrand/include -L/opt/rocm/lib/ -L/opt/rocm/hiprand/lib -lhiprand -D CUB_SORT_TYPE=float -o sort_float_rocm.exe
fi

