# CUDA Memory Allocator Test

This program was created in order to test the cuMemAlloc() on cluster compute nodes to verify certain behaviors of other (more complex) tests.

The program accepts one or more command-line arguments representing the number of bytes to be allocated using the cuMemAlloc() function — memory resident on a GPU device.  There are two special values:

- `all`: after noting the amount of memory present on the device, attempt to allocate all of it
- `N%`: where `N` is a floating-point value indicating the percentage of the total memory present in the device

Otherwise, the argument is expected to be a positive integer value.

## MPI Parallelism

The program can be built with MPI parallelism to have each MPI rank make use of a GPU device.  Available GPU devices are chosen round-robin per rank on a node:  a job with 2 nodes and 4 ranks on each node, with 4 GPUs per node, will have each of the 8 ranks mapped to a distinct GPU.  Note that this will be tempered by the `CUDA_VISIBLE_DEVICES` environment variable, if present.  In the previous 8-rank example, if `CUDA_VISIBLE_DEVICES=0` then all 4 ranks on each node will attempt to share the same GPU device.

## Build

Ensure you have CUDA and an MPI library present in your environment.  On the DARWIN cluster at UD, for example:

```
$ vpkg_devrequire cuda/11.1.1 openmpi/4.1.0
Adding package `cuda/11.1.1-455.32.00` to your environment
Adding package `openmpi/4.1.0` to your environment
```

The build is accomplished with make:

```
$ make
nvcc --forward-unknown-to-host-compiler -o cuMemAllocTest.o -c -ccbin mpicc -I/opt/shared/cuda/11.1.1-455.32.00/include -I/opt/shared/cuda/11.1.1-455.32.00/nvvm/include -I/opt/shared/cuda/11.1.1-455.32.00/extras/CUPTI/include -I/opt/shared/cuda/11.1.1-455.32.00/extras/Debugger/include -DHAVE_MPI  cuMemAllocTest.cu
mpicc  -o cuMemAllocTest cuMemAllocTest.o -L/opt/shared/cuda/11.1.1-455.32.00/lib64 -L/opt/shared/cuda/11.1.1-455.32.00/nvvm/lib64 -L/opt/shared/cuda/11.1.1-455.32.00/extras/CUPTI/lib64  -lm -lcudart -lcuda
```

## Testing

Here is an MPI run with 4 ranks on a node with 4 NVIDIA V100's present:

```
$ mpirun -np 4 ./cuMemAllocTest all 99%
[0002:0004] 0: MPI top-level initialization, 2 of 4
[0001:0004] 0: MPI top-level initialization, 1 of 4
[0003:0004] 0: MPI top-level initialization, 3 of 4
[0000:0004] 0: MPI top-level initialization, 0 of 4
[0003:0004] 0: MPI shared comm initialization, 3 of 4
[0002:0004] 0: MPI shared comm initialization, 2 of 4
[0001:0004] 0: MPI shared comm initialization, 1 of 4
[0000:0004] 0: MPI shared comm initialization, 0 of 4
[0001:0004] 0: cuInit(0) = no error (0)
[0001:0004] 0: cuDeviceGetCount(0x7fff9f6a43c8) = no error (0) :: gpuCount = 4
[0001:0004] 0: will use device at index 1
[0001:0004] 0: cuGetDevice(0x7fff9f6a43c4, 1) = no error (0)
[0001:0004] 0: cuDeviceTotalMem(0x7fff9f6a43b0, 0x7fd800000001) = no error (0) :: gpuMemSize = 34089730048
[0000:0004] 0: cuInit(0) = no error (0)
[0000:0004] 0: cuDeviceGetCount(0x7ffcb6efef38) = no error (0) :: gpuCount = 4
[0000:0004] 0: will use device at index 0
[0000:0004] 0: cuGetDevice(0x7ffcb6efef34, 0) = no error (0)
[0003:0004] 0: cuInit(0) = no error (0)
[0003:0004] 0: cuDeviceGetCount(0x7fff997e92a8) = no error (0) :: gpuCount = 4
[0003:0004] 0: will use device at index 3
[0003:0004] 0: cuGetDevice(0x7fff997e92a4, 3) = no error (0)
[0000:0004] 0: cuDeviceTotalMem(0x7ffcb6efef20, 0x7fb300000000) = no error (0) :: gpuMemSize = 34089730048
[0003:0004] 0: cuDeviceTotalMem(0x7fff997e9290, 0x7fa700000003) = no error (0) :: gpuMemSize = 34089730048
[0002:0004] 0: cuInit(0) = no error (0)
[0002:0004] 0: cuDeviceGetCount(0x7ffe56db7ab8) = no error (0) :: gpuCount = 4
[0002:0004] 0: will use device at index 2
[0002:0004] 0: cuGetDevice(0x7ffe56db7ab4, 2) = no error (0)
[0002:0004] 0: cuDeviceTotalMem(0x7ffe56db7aa0, 0x7f3600000002) = no error (0) :: gpuMemSize = 34089730048
[0001:0004] 0: cuCtxCreate(0x7fff9f6a43b8, 0, 0x7fd800000001) = no error (0)
[0001:0004] 1: selected memory size 34089730048
[0001:0004] 1: cuMemAlloc(0x7fff9f6a4398, 34089730048) = out of memory (2)
[0001:0004] 2: selected memory size 33748832747
[0001:0004] 2: cuMemAlloc(0x7fff9f6a4398, 33748832747) = no error (0)
[0001:0004] 2: cuMemFree(0x7fd03a000000) = no error (0)
[0002:0004] 0: cuCtxCreate(0x7ffe56db7aa8, 0, 0x7f3600000002) = no error (0)
[0002:0004] 1: selected memory size 34089730048
[0000:0004] 0: cuCtxCreate(0x7ffcb6efef28, 0, 0x7fb300000000) = no error (0)
[0000:0004] 1: selected memory size 34089730048
[0003:0004] 0: cuCtxCreate(0x7fff997e9298, 0, 0x7fa700000003) = no error (0)
[0003:0004] 1: selected memory size 34089730048
[0000:0004] 1: cuMemAlloc(0x7ffcb6efef08, 34089730048) = out of memory (2)
[0000:0004] 2: selected memory size 33748832747
[0003:0004] 1: cuMemAlloc(0x7fff997e9278, 34089730048) = out of memory (2)
[0003:0004] 2: selected memory size 33748832747
[0002:0004] 1: cuMemAlloc(0x7ffe56db7a88, 34089730048) = out of memory (2)
[0002:0004] 2: selected memory size 33748832747
[0000:0004] 2: cuMemAlloc(0x7ffcb6efef08, 33748832747) = no error (0)
[0003:0004] 2: cuMemAlloc(0x7fff997e9278, 33748832747) = no error (0)
[0002:0004] 2: cuMemAlloc(0x7ffe56db7a88, 33748832747) = no error (0)
[0000:0004] 2: cuMemFree(0x7fab9c000000) = no error (0)
[0003:0004] 2: cuMemFree(0x7f9f54000000) = no error (0)
[0002:0004] 2: cuMemFree(0x7f2ec4000000) = no error (0)
[0001:0004] 0: cuCtxDestroy(0x231a580) = no error (0)
[0002:0004] 0: cuCtxDestroy(0x2362820) = no error (0)
[0003:0004] 0: cuCtxDestroy(0x8e9550) = no error (0)
[0000:0004] 0: cuCtxDestroy(0x1619180) = no error (0)
```

All four devices are used, and each rank successfully allocates 99% of the nominal memory present on the GPU.

The same run but with `CUDA_VISIBLE_DEVICES` restricting the available GPU devices:

```
[frey@r2v00 cuMemAllocTest]$ CUDA_VISIBLE_DEVICES=0 mpirun -np 4 ./cuMemAllocTest 99%
[0001:0004] 0: MPI top-level initialization, 1 of 4
[0002:0004] 0: MPI top-level initialization, 2 of 4
[0003:0004] 0: MPI top-level initialization, 3 of 4
[0000:0004] 0: MPI top-level initialization, 0 of 4
[0003:0004] 0: MPI shared comm initialization, 3 of 4
[0001:0004] 0: MPI shared comm initialization, 1 of 4
[0002:0004] 0: MPI shared comm initialization, 2 of 4
[0000:0004] 0: MPI shared comm initialization, 0 of 4
[0002:0004] 0: cuInit(0) = no error (0)
[0002:0004] 0: cuDeviceGetCount(0x7ffc9f828ba8) = no error (0) :: gpuCount = 1
[0002:0004] 0: will use device at index 0
[0002:0004] 0: cuGetDevice(0x7ffc9f828ba4, 0) = no error (0)
[0002:0004] 0: cuDeviceTotalMem(0x7ffc9f828b90, 0x7f9300000000) = no error (0) :: gpuMemSize = 34089730048
[0001:0004] 0: cuInit(0) = no error (0)
[0001:0004] 0: cuDeviceGetCount(0x7ffeba5e13b8) = no error (0) :: gpuCount = 1
[0001:0004] 0: will use device at index 0
[0001:0004] 0: cuGetDevice(0x7ffeba5e13b4, 0) = no error (0)
[0001:0004] 0: cuDeviceTotalMem(0x7ffeba5e13a0, 0x7f6200000000) = no error (0) :: gpuMemSize = 34089730048
[0003:0004] 0: cuInit(0) = no error (0)
[0003:0004] 0: cuDeviceGetCount(0x7fffbf4742c8) = no error (0) :: gpuCount = 1
[0003:0004] 0: will use device at index 0
[0003:0004] 0: cuGetDevice(0x7fffbf4742c4, 0) = no error (0)
[0003:0004] 0: cuDeviceTotalMem(0x7fffbf4742b0, 0x7f6a00000000) = no error (0) :: gpuMemSize = 34089730048
[0000:0004] 0: cuInit(0) = no error (0)
[0000:0004] 0: cuDeviceGetCount(0x7ffc124f4288) = no error (0) :: gpuCount = 1
[0000:0004] 0: will use device at index 0
[0000:0004] 0: cuGetDevice(0x7ffc124f4284, 0) = no error (0)
[0000:0004] 0: cuDeviceTotalMem(0x7ffc124f4270, 0x7f5100000000) = no error (0) :: gpuMemSize = 34089730048
[0001:0004] 0: cuCtxCreate(0x7ffeba5e13a8, 0, 0x7f6200000000) = no error (0)
[0001:0004] 1: selected memory size 33748832747
[0000:0004] 0: cuCtxCreate(0x7ffc124f4278, 0, 0x7f5100000000) = no error (0)
[0000:0004] 1: selected memory size 33748832747
[0001:0004] 1: cuMemAlloc(0x7ffeba5e1388, 33748832747) = out of memory (2)
[0003:0004] 0: cuCtxCreate(0x7fffbf4742b8, 0, 0x7f6a00000000) = no error (0)
[0003:0004] 1: selected memory size 33748832747
[0000:0004] 1: cuMemAlloc(0x7ffc124f4258, 33748832747) = out of memory (2)
[0003:0004] 1: cuMemAlloc(0x7fffbf474298, 33748832747) = out of memory (2)
[0002:0004] 0: cuCtxCreate(0x7ffc9f828b98, 0, 0x7f9300000000) = no error (0)
[0002:0004] 1: selected memory size 33748832747
[0002:0004] 1: cuMemAlloc(0x7ffc9f828b78, 33748832747) = out of memory (2)
[0001:0004] 0: cuCtxDestroy(0x17673a0) = no error (0)
[0000:0004] 0: cuCtxDestroy(0x1cde210) = no error (0)
[0002:0004] 0: cuCtxDestroy(0x13e1e90) = no error (0)
[0003:0004] 0: cuCtxDestroy(0x12931f0) = no error (0)
```

None of the allocations of 99% of the nominal device memory were successful.  Dropping to 45%, for example:

```
  :
[0000:0004] 1: cuMemAlloc(0x7ffc7b2e21b8, 15340378521) = out of memory (2)
[0001:0004] 1: cuMemAlloc(0x7ffca6b067f8, 15340378521) = no error (0)
[0003:0004] 1: cuMemAlloc(0x7ffd6443d558, 15340378521) = out of memory (2)
[0002:0004] 1: cuMemAlloc(0x7fffc1b68638, 15340378521) = no error (0)
  :
```

Two of the ranks succeeded (summing to 90% of the device memory), the other two failed.
