#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#ifdef HAVE_MPI
#include <mpi.h>

#define PRINTF_FORMAT_HEADER    "[%04d:%04d] "
#define PRINTF_ARG_HEADER       splitRank, splitSize,
#else
#define PRINTF_FORMAT_HEADER
#define PRINTF_ARG_HEADER
#endif

#ifndef GPU_CONTEXT_FLAGS
#define GPU_CONTEXT_FLAGS (0)
#endif

unsigned int gpuContextFlags = GPU_CONTEXT_FLAGS;

int
main(
    int               argc,
    const char*       argv[]
)
{
    int               argi, indexWidth, topRank, topSize, splitRank, splitSize;
    int               gpuCount = 0, gpuIndex;
    CUdevice          gpuDevice;
    CUcontext         gpuContext;
    size_t            gpuMemSize;
    CUresult          rc;
    const char        *rcStr;

#ifdef HAVE_MPI
    MPI_Comm          shmcomm;
#endif

    if ( argc < 2 ) {
        fprintf(stderr, "usage:\n\n    %s <mem_size_bytes> {<mem_size_bytes> ..}\n\n", argv[0]);
        return 1;
    }
    indexWidth = 1 + floor(log10(argc - 1));

#ifdef HAVE_MPI
    MPI_Init_thread(&argc, (char***)&argv, MPI_THREAD_SERIALIZED, &argi);
    MPI_Comm_size(MPI_COMM_WORLD, &topSize); splitSize = topSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &topRank); splitRank = topRank;
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: MPI top-level initialization, %d of %d\n", PRINTF_ARG_HEADER indexWidth, 0, topRank, topSize);
    
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_size(shmcomm, &splitSize);
    MPI_Comm_rank(shmcomm, &splitRank);
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: MPI shared comm initialization, %d of %d\n", PRINTF_ARG_HEADER indexWidth, 0, splitRank, splitSize);
#endif

    rc = cuInit(0);
    cuGetErrorString(rc, &rcStr); 
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuInit(0) = %s (%d)\n", PRINTF_ARG_HEADER indexWidth, 0, rcStr, rc);
    if ( rc != CUDA_SUCCESS ) return 1;

    rc = cuDeviceGetCount(&gpuCount);
    cuGetErrorString(rc, &rcStr);
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuDeviceGetCount(%p) = %s (%d) :: gpuCount = %d\n", PRINTF_ARG_HEADER indexWidth, 0, &gpuCount, rcStr, rc, gpuCount);
    if ( (rc != CUDA_SUCCESS) || (gpuCount < 1) ) return 1;
    
    gpuIndex = splitRank % gpuCount;
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: will use device at index %d\n", PRINTF_ARG_HEADER indexWidth, 0, gpuIndex);
    
    rc = cuDeviceGet(&gpuDevice, gpuIndex);
    cuGetErrorString(rc, &rcStr);
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuGetDevice(%p, %d) = %s (%d)\n", PRINTF_ARG_HEADER indexWidth, 0, &gpuDevice, gpuIndex, rcStr, rc);
    if ( rc != CUDA_SUCCESS ) return 1;

    rc = cuDeviceTotalMem(&gpuMemSize, gpuDevice);
    cuGetErrorString(rc, &rcStr);
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuDeviceTotalMem(%p, %p) = %s (%d) :: gpuMemSize = %lld\n", PRINTF_ARG_HEADER indexWidth, 0, &gpuMemSize, gpuDevice, rcStr, rc, (long long)gpuMemSize);
    if ( rc != CUDA_SUCCESS ) return 1;

    rc = cuCtxCreate(&gpuContext, gpuContextFlags, gpuDevice);
    cuGetErrorString(rc, &rcStr);
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuCtxCreate(%p, %x, %p) = %s (%d)\n", PRINTF_ARG_HEADER indexWidth, 0, &gpuContext, gpuContextFlags, gpuDevice, rcStr, rc);
    if ( rc != CUDA_SUCCESS ) return 1;

    argi = 1;
    while ( argi < argc ) {
        CUdeviceptr   gpuMem;
        long          reqMemSize;
        char          *endptr;

        if ( strchr(argv[argi], '%') ) {
            double    percentage = strtod(argv[argi], &endptr);

            if ( (percentage <= 0.0) || (endptr == argv[argi]) ) {
                fprintf(stderr, "ERROR:  invalid memory percentage: %s\n", argv[argi]);
                return errno;
            }
            reqMemSize = floor(gpuMemSize * 0.01 * percentage);
        } else if ( strcmp(argv[argi], "all") == 0 ) {
            reqMemSize = gpuMemSize;
        } else {
            reqMemSize = strtol(argv[argi], &endptr, 0);
            if ( (reqMemSize == 0) && (endptr == argv[argi]) ) {
                fprintf(stderr, "ERROR:  invalid memory size: %s\n", argv[argi]);
                return errno;
            }
            if ( reqMemSize == 0 ) reqMemSize = gpuMemSize;
            else if ( reqMemSize < 0 ) reqMemSize = floor((1.0 - 0.125) * gpuMemSize);
        }
        fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: selected memory size %ld\n", PRINTF_ARG_HEADER indexWidth, argi, reqMemSize);
        rc = cuMemAlloc(&gpuMem, (size_t)reqMemSize);
        cuGetErrorString(rc, &rcStr);
        fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuMemAlloc(%p, %ld) = %s (%d)\n", PRINTF_ARG_HEADER indexWidth, argi, &gpuMem, reqMemSize, rcStr, rc);
        if ( rc == CUDA_SUCCESS) {
            rc = cuMemFree(gpuMem);
            cuGetErrorString(rc, &rcStr);
            fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuMemFree(%p) = %s (%d)\n", PRINTF_ARG_HEADER indexWidth, argi, gpuMem, rcStr, rc);
        }
        argi++;
    }

    rc = cuCtxDestroy(gpuContext);
    cuGetErrorString(rc, &rcStr);
    fprintf(stderr, PRINTF_FORMAT_HEADER "%0*d: cuCtxDestroy(%p) = %s (%d)\n", PRINTF_ARG_HEADER indexWidth, 0, gpuContext, rcStr, rc);
    if ( rc != CUDA_SUCCESS ) return 1;

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}

