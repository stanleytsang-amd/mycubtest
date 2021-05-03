#include "float.h"
#include <hiprand.h>
#include <hipcub/hipcub.hpp>

//
//
//

#include <stdbool.h>

static
void
hip_assert(const hipError_t code, const char* const file, const int line, const bool abort)
{
  if (code != hipSuccess)
    {
      fprintf(stderr,"hip_assert: %s %s %d\n",hipGetErrorString(code),file,line);

      if (abort)
        {
          hipDeviceReset();
          exit(code);
        }
    }
}

#define hip(...) { hip_assert((hip##__VA_ARGS__), __FILE__, __LINE__, true); }

//
//
//

#ifndef CUB_SORT_TYPE
#define CUB_SORT_TYPE   uint64_t
#endif

#define CUB_SORT_WARMUP 100
#define CUB_SORT_BENCH  100

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
  hip(EventRecord(start,0));

  hipcub::DeviceRadixSort::SortKeys(tmp,tmp_size,vin_d,vout_d,count);

  hip(EventRecord(end,0));
  hip(EventSynchronize(end));

  float t_ms;
  hip(EventElapsedTime(&t_ms,start,end));


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
bench(const hipDeviceProp_t* const props, const uint32_t count)
{
  //
  // allocate
  //
  size_t const    vin_size = sizeof(CUB_SORT_TYPE) * count;
  CUB_SORT_TYPE * vin_d;
  CUB_SORT_TYPE * vout_d;

  hip(Malloc(&vin_d, vin_size));
  hip(Malloc(&vout_d,vin_size));

  //
  // fill with random values
  //
  hiprandGenerator_t prng;

  hiprandCreateGenerator(&prng,HIPRAND_RNG_PSEUDO_XORWOW);

  hiprandSetPseudoRandomGeneratorSeed(prng,0xCAFEBABE);

  if      (sizeof(CUB_SORT_TYPE) == sizeof(unsigned int)) {
    hiprandGenerate(prng,(unsigned int*)vin_d,count);
  } else if (sizeof(CUB_SORT_TYPE) == sizeof(unsigned long long)) {
    hiprandGenerateUniform(prng,(float*)vin_d,count);
  } else {
    exit(EXIT_FAILURE);
  }

  //
  // size and allocate the temp array
  //
  void * tmp;
  size_t tmp_size = 0;

  hipcub::DeviceRadixSort::SortKeys(NULL,tmp_size,vin_d,vout_d,count);
  hip(Malloc(&tmp,tmp_size));

  //
  // benchmark
  //
  hipEvent_t start, end;
  hip(EventCreate(&start));
  hip(EventCreate(&end));

  float min_ms     = 999;//FLT_MAX;
  float max_ms     = 0.0f;
  float elapsed_ms = 0.0f;

  for (int ii=0; ii<CUB_SORT_WARMUP; ii++)
    sort(count,vin_d,vout_d,tmp,tmp_size,start,end,
         &min_ms,
         &max_ms,
         &elapsed_ms);

  min_ms     = 999;//FLT_MAX;
  max_ms     = 0.0f;
  elapsed_ms = 0.0f;

  for (int ii=0; ii<CUB_SORT_BENCH; ii++)
    sort(count,vin_d,vout_d,tmp,tmp_size,start,end,
         &min_ms,
         &max_ms,
         &elapsed_ms);

  hip(EventDestroy(start));
  hip(EventDestroy(end));

  //
  //
  //
  hip(Free(tmp));
  hip(Free(vout_d));
  hip(Free(vin_d));

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

  hip(SetDevice(device));
  hipDeviceProp_t props;
  hip(GetDeviceProperties(&props,device));

  //printf("%s (%2d)\n",props.name,props.multiProcessorCount);


  //
  //
  //
  const uint32_t count_lo   = argc <= 2 ? 2048   : strtoul(argv[2],NULL,0);
  const uint32_t count_hi   = argc <= 3 ? 262144 : strtoul(argv[3],NULL,0);
  const uint32_t count_step = argc <= 4 ? 2048   : strtoul(argv[4],NULL,0);

  //
  // LABELS
  //
//  fprintf(stdout,
//          "Device, "
//          "Multiprocessors, "
//          "Type, "
//          "Keys, "
//          "Trials, "
//          "Total Msecs, "
//          "Avg. Msecs, "
//          "Min Msecs, "
//          "Max Msecs, "
//          "Avg. Mkeys/s, "
//          "Max. Mkeys/s\n");

  //
  // SORT
  //
  for (uint32_t count=count_lo; count<=count_hi; count+=count_step)
    bench(&props,count);

  //
  // RESET
  //
  hip(DeviceReset());

  return 0;
}
