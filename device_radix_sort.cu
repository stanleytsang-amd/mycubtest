#include <curand.h>
#include <cub/cub.cuh>

//
//
//

#include <stdbool.h>

static
void
cuda_assert(const cudaError_t code, const char* const file, const int line, const bool abort)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"cuda_assert: %s %s %d\n",cudaGetErrorString(code),file,line);

      if (abort)
        {
          cudaDeviceReset();
          exit(code);
        }
    }
}

#define cuda(...) { cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true); }

//
//
//

#ifndef CUB_SORT_TYPE
#define CUB_SORT_TYPE   uint64_t
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
     cudaEvent_t     start,
     cudaEvent_t     end,
     float         * min_ms,
     float         * max_ms,
     float         * elapsed_ms)
{
  cuda(EventRecord(start,0));

  cub::DeviceRadixSort::SortKeys(tmp,tmp_size,vin_d,vout_d,count);

  cuda(EventRecord(end,0));
  cuda(EventSynchronize(end));

  float t_ms;
  cuda(EventElapsedTime(&t_ms,start,end));

  *min_ms      = min(*min_ms,t_ms);
  *max_ms      = max(*max_ms,t_ms);
  *elapsed_ms += t_ms;
}

//
//
//

static
void
bench(const struct cudaDeviceProp* const props, const uint32_t count)
{
  //
  // allocate
  //
  size_t const    vin_size = sizeof(CUB_SORT_TYPE) * count;
  CUB_SORT_TYPE * vin_d;
  CUB_SORT_TYPE * vout_d;

  cuda(Malloc(&vin_d, vin_size));
  cuda(Malloc(&vout_d,vin_size));

  //
  // fill with random values
  //
  curandGenerator_t prng;

  curandCreateGenerator(&prng,CURAND_RNG_PSEUDO_XORWOW);

  curandSetPseudoRandomGeneratorSeed(prng,0xCAFEBABE);

  if      (sizeof(CUB_SORT_TYPE) == sizeof(unsigned int)) {
    curandGenerate(prng,(unsigned int*)vin_d,count);
  } else if (sizeof(CUB_SORT_TYPE) == sizeof(unsigned long long)) {
    curandGenerateLongLong(prng,(unsigned long long*)vin_d,count);
  } else {
    exit(EXIT_FAILURE);
  }

  //
  // size and allocate the temp array
  //
  void * tmp;
  size_t tmp_size = 0;

  cub::DeviceRadixSort::SortKeys(NULL,tmp_size,vin_d,vout_d,count);
  cuda(Malloc(&tmp,tmp_size));

  //
  // benchmark
  //
  cudaEvent_t start, end;
  cuda(EventCreate(&start));
  cuda(EventCreate(&end));

  float min_ms     = FLT_MAX;
  float max_ms     = 0.0f;
  float elapsed_ms = 0.0f;

  for (int ii=0; ii<CUB_SORT_WARMUP; ii++)
    sort(count,vin_d,vout_d,tmp,tmp_size,start,end,
         &min_ms,
         &max_ms,
         &elapsed_ms);

  min_ms     = FLT_MAX;
  max_ms     = 0.0f;
  elapsed_ms = 0.0f;

  for (int ii=0; ii<CUB_SORT_BENCH; ii++)
    sort(count,vin_d,vout_d,tmp,tmp_size,start,end,
         &min_ms,
         &max_ms,
         &elapsed_ms);

  cuda(EventDestroy(start));
  cuda(EventDestroy(end));

  //
  //
  //
  cuda(Free(tmp));
  cuda(Free(vout_d));
  cuda(Free(vin_d));

  //
  //
  //
#define STRINGIFY2(s) #s
#define STRINGIFY(s)  STRINGIFY2(s)

  fprintf(stdout,"%s, %u, %s, %u, %u, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
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
}

//
//
//

int
main(int argc, char** argv)
{
  const int32_t device = (argc == 1) ? 0 : atoi(argv[1]);

  struct cudaDeviceProp props;
  cuda(GetDeviceProperties(&props,device));

  printf("%s (%2d)\n",props.name,props.multiProcessorCount);

  cuda(SetDevice(device));

  //
  //
  //
  const uint32_t count_lo   = argc <= 2 ? 2048   : strtoul(argv[2],NULL,0);
  const uint32_t count_hi   = argc <= 3 ? 262144 : strtoul(argv[3],NULL,0);
  const uint32_t count_step = argc <= 4 ? 2048   : strtoul(argv[4],NULL,0);

  //
  // LABELS
  //
  fprintf(stdout,
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
  for (uint32_t count=count_lo; count<=count_hi; count+=count_step)
    bench(&props,count);

  //
  // RESET
  //
  cuda(DeviceReset());

  return 0;
}
