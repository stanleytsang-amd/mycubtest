#include "float.h"
#include <hiprand.h>
#include <hipcub/hipcub.hpp>
#include <time.h>
#include <fstream>
#include <string>
//
//
//

#include <stdbool.h>

//static
//void
//hip_assert(const hipError_t code, const char* const file, const int line, const bool abort)
//{
//  if (code != hipSuccess)
//    {
//      fprintf(stderr,"hip_assert: %s %s %d\n",hipGetErrorString(code),file,line);
//
//      if (abort)
//        {
//          hipDeviceReset();
//          exit(code);
//        }
//    }
//}
//
//#define hip(...) { hip_assert((hip##__VA_ARGS__), __FILE__, __LINE__, true); }



#define DEBUG_HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( hipError_t err, const char *file, int line )
{

    if (err != hipSuccess)
    {
    	fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
						hipGetErrorString( err ), file, line, err );
    	    fflush(stdout);
            exit(err);
    }
}



//
//
//

#ifndef CUB_SORT_TYPE
#define CUB_SORT_TYPE   float
#endif

#define CUB_SORT_WARMUP 10
#define CUB_SORT_BENCH  10

//
//
//

static
void
sort(uint32_t        count,
     CUB_SORT_TYPE * vin_d,
     CUB_SORT_TYPE * vout_d,
     void  *         tmp,
     size_t        & tmp_size,
     hipEvent_t     start,
     hipEvent_t     end,
     float         * min_ms,
     float         * max_ms,
     float         * elapsed_ms)
{
  DEBUG_HANDLE_ERROR(hipEventRecord(start,0));

  hipcub::DeviceRadixSort::SortKeys(tmp,tmp_size,vin_d,vout_d,count,0, sizeof(float)*8,0);
  //hipcub::DeviceRadixSort::SortKeys(tmp,tmp_size,vin_d,vout_d,count);

  DEBUG_HANDLE_ERROR(hipEventRecord(end,0));
  DEBUG_HANDLE_ERROR(hipEventSynchronize(end));

  float t_ms;
  DEBUG_HANDLE_ERROR(hipEventElapsedTime(&t_ms,start,end));


//  *min_ms      = (float)min(*min_ms,t_ms);
  if (t_ms < *min_ms)
    *min_ms = t_ms;
//  *max_ms      = (float)max(*max_ms,t_ms);
  if (t_ms > *max_ms)
    *max_ms = t_ms;
  *elapsed_ms += t_ms;
}

//
//
//

static
void
bench(FILE *fp, const hipDeviceProp_t* const props, const uint32_t count)
{
  clock_t start, end;
  double cpu_time_used;
  //
  // allocate
  //
  size_t const    vin_size = sizeof(CUB_SORT_TYPE) * count;
  CUB_SORT_TYPE * vin_d;
  CUB_SORT_TYPE * vout_d;

  start = clock();
  DEBUG_HANDLE_ERROR(hipMalloc(&vin_d, vin_size));
  DEBUG_HANDLE_ERROR(hipMalloc(&vout_d,vin_size));
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("Time 'allocate' = %f\n", cpu_time_used);

  //
  // fill with random values
  //
  hiprandGenerator_t prng;

  start = clock();
  hiprandCreateGenerator(&prng,HIPRAND_RNG_PSEUDO_XORWOW);

  hiprandSetPseudoRandomGeneratorSeed(prng,0xCAFEBABE);

//  if      (sizeof(CUB_SORT_TYPE) == sizeof(unsigned int)) {
//    hiprandGenerate(prng,(unsigned int*)vin_d,count);
//  } else if (sizeof(CUB_SORT_TYPE) == sizeof(unsigned long long)) {
    hiprandGenerateUniform(prng,(float*)vin_d,count);
//  } else {
//    exit(EXIT_FAILURE);
//  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("Time 'fill' = %f\n", cpu_time_used);

  //
  // size and allocate the temp array
  //
  void * tmp;
  size_t tmp_size = 0;

  hipcub::DeviceRadixSort::SortKeys(NULL,tmp_size,vin_d,vout_d,count);
  if(tmp_size==0)
      tmp_size=1;

  DEBUG_HANDLE_ERROR(hipMalloc((void**)&tmp,tmp_size));

  //
  // benchmark
  //
  hipEvent_t _start, _end;
  DEBUG_HANDLE_ERROR(hipEventCreate(&_start));
  DEBUG_HANDLE_ERROR(hipEventCreate(&_end));

  float min_ms     = 999;//FLT_MAX;
  float max_ms     = 0.0f;
  float elapsed_ms = 0.0f;

  start = clock();
  for (int ii=0; ii<CUB_SORT_WARMUP; ii++) {
    sort(count,vin_d,vout_d,tmp,tmp_size,_start,_end,
         &min_ms,
         &max_ms,
         &elapsed_ms);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("Time 'warmup' = %f\n", cpu_time_used);

  min_ms     = 999;//FLT_MAX;
  max_ms     = 0.0f;
  elapsed_ms = 0.0f;

  start = clock();
  for (int ii=0; ii<CUB_SORT_BENCH; ii++) {
    sort(count,vin_d,vout_d,tmp,tmp_size,_start,_end,
         &min_ms,
         &max_ms,
         &elapsed_ms);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("Time 'benchmark' = %f\n", cpu_time_used);

  DEBUG_HANDLE_ERROR(hipEventDestroy(_start));
  DEBUG_HANDLE_ERROR(hipEventDestroy(_end));

  //
  //
  //
  hiprandDestroyGenerator(prng);
  DEBUG_HANDLE_ERROR(hipFree(tmp));
  DEBUG_HANDLE_ERROR(hipFree(vout_d));
  DEBUG_HANDLE_ERROR(hipFree(vin_d));


  //
  //
  //
#define STRINGIFY2(s) #s
#define STRINGIFY(s)  STRINGIFY2(s)

  start = clock();
  fprintf(fp/*stdout*/,"%s, %u, %s, %u, %u, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
          props->name,
          props->multiProcessorCount,
          STRINGIFY(CUB_SORT_TYPE),
          count,
          CUB_SORT_BENCH,
          elapsed_ms,
          (double)elapsed_ms / CUB_SORT_BENCH,
          (double)min_ms,
          (double)max_ms,
          (double)(CUB_SORT_BENCH * count) / (1000.0 * elapsed_ms),
          (double)count                    / (1000.0 * min_ms));
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("Time 'print' = %f\n", cpu_time_used);

//  fstream outFile;
//  outFile.open("rocm-0504.csv", ios::app);
//
//  // Write to the file
//  outFile << ;
//
//  // Close file
//  outFile.close();

}

//
//
//

int
main(int argc, char** argv)
{
  const int32_t device = (argc == 1) ? 1 : atoi(argv[1]);

  DEBUG_HANDLE_ERROR(hipSetDevice(device));
  hipDeviceProp_t props;
  DEBUG_HANDLE_ERROR(hipGetDeviceProperties(&props,device));

  //printf("%s (%2d)\n",props.name,props.multiProcessorCount);

  FILE *fp;  
  fp = fopen("rocm-0504-2.csv", "w");

  //
  //
  //
  const uint32_t count_lo   = argc <= 2 ? 2048   : strtoul(argv[2],NULL,0);
  const uint32_t count_hi   = argc <= 3 ? 262144 : strtoul(argv[3],NULL,0);
  const uint32_t count_step = argc <= 4 ? 2048   : strtoul(argv[4],NULL,0);

  //
  // LABELS
  //
  fprintf(fp/*stdout*/,
          "Device, "
          "Multiprocessors, "
          "Type, "
          "Keys, "
          "Trials, "
          "Total Msecs, "
          "Avg. Msecs, "
          "Min Msecs, "
          "Max Msecs, "
          "Avg. Mkeys/s, "
          "Max. Mkeys/s\n");
  //
  // SORT
  //

  long long count = 0;
  std::ifstream file("iter2-25-cub.size.sort.uniq.reverse");
  //std::ifstream file("test.input");
  std::string str;
  while (std::getline(file, str)) {
    if (count % 1000 == 0) {
        std::cout << "COUNT: " << count << "\n";
    }
    count++;
    bench(fp, &props, std::stoi(str));
  }
  fclose(fp);

  //
  // RESET
  //
  DEBUG_HANDLE_ERROR(hipDeviceReset());

  return 0;
}
