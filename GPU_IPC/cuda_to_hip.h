//
// cuda_to_hip.h
// CUDA-to-HIP compatibility header for GPU_IPC ROCm port
//
// This header provides CUDA-to-HIP symbol mappings when building with HIP.
// On CUDA, it simply includes the standard CUDA runtime header.
//

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// Memory management
#define cudaMalloc                hipMalloc
#define cudaFree                  hipFree
#define cudaMemcpy                hipMemcpy
#define cudaMemcpyAsync           hipMemcpyAsync
#define cudaMemset                hipMemset
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice

// Device management
#define cudaSetDevice             hipSetDevice
#define cudaDeviceSynchronize     hipDeviceSynchronize
#define cudaDeviceReset           hipDeviceReset

// Error handling
#define cudaError_t               hipError_t
#define cudaError                 hipError_t
#define cudaSuccess               hipSuccess
#define cudaGetErrorString        hipGetErrorString

// Events
#define cudaEvent_t               hipEvent_t
#define cudaEventCreate           hipEventCreate
#define cudaEventRecord           hipEventRecord
#define cudaEventDestroy          hipEventDestroy
#define cudaEventSynchronize      hipEventSynchronize
#define cudaEventElapsedTime      hipEventElapsedTime

// Streams
#define cudaStream_t              hipStream_t
#define cudaStreamCreate          hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy         hipStreamDestroy
#define cudaStreamSynchronize     hipStreamSynchronize
#define cudaStreamNonBlocking     hipStreamNonBlocking

// Warp intrinsics. A CUDA warp is always 32 lanes, so a *_sync form with a full member
// mask acts on exactly 32 lanes and its ballot is a 32-bit mask. An AMD wavefront is 32
// lanes on RDNA and 64 on CDNA, and the unqualified HIP intrinsics take the wavefront
// width. The faithful translation therefore pins the shuffle width to 32 and shifts a
// wavefront-wide ballot down to the calling lane's own 32-lane half: that is the identity
// on a 32-lane wavefront and selects the correct half on a 64-lane one. Truncating the
// ballot instead would hand the upper half of a 64-lane wavefront the lower half's bits.
// Guarded because the host compiler also sees this header for the plain C++ sources, and
// the wavefront intrinsics only exist in the HIP compiler's translation units.
#if defined(__HIPCC__)
__device__ inline unsigned int hip_ballot_warp32(int predicate)
{
    unsigned long long mask = __ballot(predicate);
    return (unsigned int)(mask >> (__lane_id() & ~31u));
}
#endif

#define __ballot_sync(mask, predicate)     hip_ballot_warp32(predicate)
#define __shfl_sync(mask, var, srcLane)    __shfl(var, srcLane, 32)
#define __shfl_down_sync(mask, var, delta) __shfl_down(var, delta, 32)
#define __shfl_up_sync(mask, var, delta)   __shfl_up(var, delta, 32)
#define __shfl_xor_sync(mask, var, mask2)  __shfl_xor(var, mask2, 32)

#else

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#endif
