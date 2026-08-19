__device__ __forceinline__ void SYNC_THREADS()
{
    __syncthreads();
}

__device__ __forceinline__ void THREAD_FENCE()
{
    __threadfence();
}

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
// A CUDA warp is always 32 lanes, so a *_sync form with a full member mask acts on
// exactly 32 lanes and its ballot is a 32-bit mask. An AMD wavefront is 32 lanes on
// RDNA and 64 on CDNA, and the unqualified HIP intrinsics take the wavefront width.
// The faithful translation therefore pins the shuffle width to 32 and shifts a
// wavefront-wide ballot down to the calling lane's own 32-lane half: that is the
// identity on a 32-lane wavefront and selects the correct half on a 64-lane one.
// Truncating the ballot instead would hand the upper half of a 64-lane wavefront the
// lower half's bits.
__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate, unsigned int member_mask = 0xffffffff)
{
    (void)member_mask;
    unsigned long long mask = __ballot(predicate);
    return (unsigned int)(mask >> (__lane_id() & ~31u));
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL(Type var, int srcLane, unsigned int member_mask = 0xffffffff)
{
    (void)member_mask;
    return __shfl(var, srcLane, 32);
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL_DOWN(Type var, unsigned int delta, unsigned int member_mask = 0xffffffff)
{
    (void)member_mask;
    return __shfl_down(var, delta, 32);
}
#else
__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate, unsigned int member_mask = 0xffffffff)
{
    return __ballot_sync(member_mask, predicate);
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL(Type var, int srcLane, unsigned int member_mask = 0xffffffff)
{
    return __shfl_sync(member_mask, var, srcLane);
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL_DOWN(Type var, unsigned int delta, unsigned int member_mask = 0xffffffff)
{
    return __shfl_down_sync(member_mask, var, delta);
}
#endif


template <class TypeA, class TypeB>
__device__ __forceinline__ TypeA ATOMIC_ADD(TypeA* dest, const TypeB& source)
{
    return atomicAdd(dest, source);
}