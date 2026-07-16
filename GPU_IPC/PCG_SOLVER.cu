//
// PCG_SOLVER.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//
// GPU preconditioned conjugate gradient solver.
//
// File layout:
//   1. Block-reduction device helper (shared by all reduction kernels)
//   2. Reduction kernels (dot products, norms, preconditioned variants)
//   3. Element-wise vector-update kernels
//   4. Fused Hessian SpMV kernel        (__PCG_Solve_AXALL_b2)
//   5. Block-Jacobi preconditioner assembly kernels
//   6. Host-side reduction drivers
//   7. Host-side kernel launch wrappers
//   8. PCG / MASPCG main loops
//   9. PCG_Data / BHessian device-memory management
//

#include "PCG_SOLVER.cuh"
#include "device_launch_parameters.h"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools.h"
#include "device_utils.h"

// =============================================================================
// 1. Block-reduction device helper
// =============================================================================

// Warp-shuffle reduction of `temp` across the whole block; the per-block
// partial sum is written to squeue[blockIdx.x]. `numbers` is the number of
// active elements in the (possibly partial) last block.
//
// This is the exact epilogue previously copy-pasted into every reduction
// kernel below; behavior is unchanged.
__device__ __forceinline__ void __PCG_blockReducePartial(double temp, double* tep, double* squeue, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += gipc::WARP_SHFL_DOWN(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    gipc::SYNC_THREADS();
    if (threadIdx.x < warpNum) {
        if (warpNum > 1) {
            temp = tep[threadIdx.x];
            for (int i = 1; i < warpNum; i = (i << 1)) {
                temp += gipc::WARP_SHFL_DOWN(temp, i);
            }
        }
        if (threadIdx.x == 0) {
            squeue[blockIdx.x] = temp;
        }
    }
}

// =============================================================================
// 2. Reduction kernels
// =============================================================================

// Partial dot product: squeue[block] = sum_i a[i] . b[i]
__global__ void PCG_vdv_Reduction(double* squeue, const double3* a, const double3* b, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    double temp = __GEIGEN__::__v_vec_dot(a[idx], b[idx]);

    __PCG_blockReducePartial(temp, tep, squeue, numbers);
}

// Second-stage reduction: squeue[block] = sum_i squeue[i]
__global__ void add_reduction(double* mem, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    double temp = mem[idx];

    gipc::THREAD_FENCE();

    __PCG_blockReducePartial(temp, tep, mem, numbers);
}

// delta0 = sum_i (C*b)_i^T * P_i * (C*b)_i  (energy norm of the filtered RHS)
__global__ void PCG_add_Reduction_delta0(double* squeue, const __GEIGEN__::Matrix3x3d* P, const double3* b, const __GEIGEN__::Matrix3x3d* constraint, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    double3 t_b = b[idx];
    __GEIGEN__::Matrix3x3d t_constraint = constraint[idx];
    double3 filter_b = __GEIGEN__::__M_v_multiply(t_constraint, t_b);

    double temp = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_M_multiply(filter_b, P[idx]), filter_b);

    __PCG_blockReducePartial(temp, tep, squeue, numbers);
}

// deltaN0: r = C*(b - r), c = C*(P*r); squeue[block] = sum_i r_i . c_i
__global__ void PCG_add_Reduction_deltaN0(double* squeue, const __GEIGEN__::Matrix3x3d* P, const double3* b, double3* r, double3* c, const __GEIGEN__::Matrix3x3d* constraint, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    double3 t_b = b[idx];
    __GEIGEN__::Matrix3x3d t_constraint = constraint[idx];
    double3 t_r = __GEIGEN__::__M_v_multiply(t_constraint, __GEIGEN__::__minus(t_b, r[idx]));
    double3 t_c = __GEIGEN__::__M_v_multiply(P[idx], t_r);
    t_c = __GEIGEN__::__M_v_multiply(t_constraint, t_c);
    r[idx] = t_r;
    c[idx] = t_c;

    double temp = __GEIGEN__::__v_vec_dot(t_r, t_c);

    __PCG_blockReducePartial(temp, tep, squeue, numbers);
}

// deltaN: dx += alpha*c, r -= alpha*q, s = P*r; squeue[block] = sum_i r_i . s_i
__global__ void PCG_add_Reduction_deltaN(double* squeue, double3* dx, const double3* c, double3* r, const double3* q, const __GEIGEN__::Matrix3x3d* P, double3* s, double alpha, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    double3 t_c = c[idx];
    double3 t_dx = dx[idx];
    double3 t_r = r[idx];
    double3 t_q = q[idx];

    dx[idx] = __GEIGEN__::__add(t_dx, __GEIGEN__::__s_vec_multiply(t_c, alpha));
    t_r = __GEIGEN__::__add(t_r, __GEIGEN__::__s_vec_multiply(t_q, -alpha));
    r[idx] = t_r;
    double3 t_s = __GEIGEN__::__M_v_multiply(P[idx], t_r);
    s[idx] = t_s;

    double temp = __GEIGEN__::__v_vec_dot(t_r, t_s);

    __PCG_blockReducePartial(temp, tep, squeue, numbers);
}

// tempSum: q = C*q; squeue[block] = sum_i q_i . c_i
__global__ void PCG_add_Reduction_tempSum(double* squeue, const double3* c, double3* q, const __GEIGEN__::Matrix3x3d* constraint, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    double3 t_c = c[idx];
    double3 t_q = q[idx];
    __GEIGEN__::Matrix3x3d t_constraint = constraint[idx];
    t_q = __GEIGEN__::__M_v_multiply(t_constraint, t_q);
    q[idx] = t_q;

    double temp = __GEIGEN__::__v_vec_dot(t_q, t_c);

    __PCG_blockReducePartial(temp, tep, squeue, numbers);
}

// =============================================================================
// 3. Element-wise vector-update kernels
// =============================================================================

// q = mass .* c (diagonal mass part of A*x)
__global__ void __PCG_Solve_AX_mass_b(const double* _masses, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    q[idx] = __GEIGEN__::__s_vec_multiply(c[idx], _masses[idx]);
}

// dx += rate * c; r -= rate * q
__global__ void __PCG_Update_Dx_R(const double3* c, double3* dx, const double3* q, double3* r, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    dx[idx] = __GEIGEN__::__add(dx[idx], __GEIGEN__::__s_vec_multiply(c[idx], rate));
    r[idx] = __GEIGEN__::__add(r[idx], __GEIGEN__::__s_vec_multiply(q[idx], -rate));
}

// c = C * (s + rate * c)
__global__ void __PCG_FinalStep_UpdateC(const __GEIGEN__::Matrix3x3d* constraints, const double3* s, double3* c, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double3 tempc = __GEIGEN__::__add(s[idx], __GEIGEN__::__s_vec_multiply(c[idx], rate));
    c[idx] = __GEIGEN__::__M_v_multiply(constraints[idx], tempc);
}

// output = C * input (apply the per-vertex constraint filter)
__global__ void __PCG_constraintFilter(const __GEIGEN__::Matrix3x3d* constraints, const double3* input, double3* output, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    output[idx] = __GEIGEN__::__M_v_multiply(constraints[idx], input[idx]);
}

// =============================================================================
// 4. Fused Hessian SpMV kernel
// =============================================================================

// q += H * c for all Hessian block sizes in a single launch.
// Each 12x12 / 9x9 / 6x6 block is processed by 144 / 81 / 36 threads
// (one per matrix element) followed by a segmented warp reduction;
// each 3x3 block is handled by a single thread.
// The block range is partitioned by offset4/offset3/offset2.
__global__ void __PCG_Solve_AXALL_b2(const __GEIGEN__::Matrix12x12d* Hessians12, const __GEIGEN__::Matrix9x9d* Hessians9,
    const __GEIGEN__::Matrix6x6d* Hessians6, const __GEIGEN__::Matrix3x3d* Hessians3, const uint4* D4Index, const uint3* D3Index,
    const uint2* D2Index, const uint32_t* D1Index, const double3* c, double3* q, int numbers4, int numbers3, int numbers2, int numbers1,
    int offset4, int offset3, int offset2) {

    if (blockIdx.x < offset4) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numbers4) return;
        __shared__ int offset;
        int Hid = idx / 144;
        int MRid = (idx % 144) / 12;
        int MCid = (idx % 144) % 12;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 12;

        double rdata = Hessians12[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (12 - GRtid);
        }
        gipc::SYNC_THREADS();

        int BRid = (threadIdx.x - offset + 12) / 12;
        int landidx = (threadIdx.x - offset) % 12;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = gipc::WARP_BALLOT(bBoundary);
        mark = __brev(mark);
        unsigned int interval = std::min<unsigned int>(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 12; iter <<= 1) {
            double tmp = gipc::WARP_SHFL_DOWN(rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            gipc::ATOMIC_ADD((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 && blockIdx.x < offset4 + offset3) {
        int idx = (blockIdx.x - offset4) * blockDim.x + threadIdx.x;
        if (idx >= numbers3) return;
        __shared__ int offset;
        int Hid = idx / 81;
        int MRid = (idx % 81) / 9;
        int MCid = (idx % 81) % 9;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 9;

        double rdata = Hessians9[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (9 - GRtid);
        }
        gipc::SYNC_THREADS();

        int BRid = (threadIdx.x - offset + 9) / 9;
        int landidx = (threadIdx.x - offset) % 9;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = gipc::WARP_BALLOT(bBoundary); // a bit-mask
        mark = __brev(mark);
        unsigned int interval =
            std::min<unsigned int>(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 9; iter <<= 1) {
            double tmp = gipc::WARP_SHFL_DOWN(rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            gipc::ATOMIC_ADD((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 + offset3 && blockIdx.x < offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3) * blockDim.x + threadIdx.x;
        if (idx >= numbers2) return;
        __shared__ int offset;
        int Hid = idx / 36;
        int MRid = (idx % 36) / 6;
        int MCid = (idx % 36) % 6;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 6;

        double rdata = Hessians6[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (6 - GRtid);
        }
        gipc::SYNC_THREADS();

        int BRid = (threadIdx.x - offset + 6) / 6;
        int landidx = (threadIdx.x - offset) % 6;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = gipc::WARP_BALLOT(bBoundary);
        mark = __brev(mark);
        unsigned int interval =
            std::min<unsigned int>(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 6; iter <<= 1) {
            double tmp = gipc::WARP_SHFL_DOWN(rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            gipc::ATOMIC_ADD((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3 - offset2) * blockDim.x + threadIdx.x;
        if (idx >= numbers1) return;
        __GEIGEN__::Matrix3x3d H = Hessians3[idx];
        double3 tempC, tempQ;

        tempC.x = c[D1Index[idx]].x;
        tempC.y = c[D1Index[idx]].y;
        tempC.z = c[D1Index[idx]].z;


        tempQ = __GEIGEN__::__M_v_multiply(H, tempC);

        gipc::ATOMIC_ADD(&(q[D1Index[idx]].x), tempQ.x);
        gipc::ATOMIC_ADD(&(q[D1Index[idx]].y), tempQ.y);
        gipc::ATOMIC_ADD(&(q[D1Index[idx]].z), tempQ.z);
    }
}

// =============================================================================
// 5. Block-Jacobi preconditioner assembly kernels
// =============================================================================

// Accumulate the 3x3 diagonal blocks of every Hessian block into P
// (atomic because vertices are shared between blocks).
__global__ void __PCG_AXALL_P(const __GEIGEN__::Matrix12x12d* Hessians12, const __GEIGEN__::Matrix9x9d* Hessians9,
    const __GEIGEN__::Matrix6x6d* Hessians6, const __GEIGEN__::Matrix3x3d* Hessians3,
    const uint4* D4Index, const uint3* D3Index, const uint2* D2Index, const uint32_t* D1Index,
    __GEIGEN__::Matrix3x3d* P, int numbers4, int numbers3, int numbers2, int numbers1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers4 + numbers3 + numbers2 + numbers1) return;

    if (idx < numbers4) {
        int Hid = idx / 12;
        int qid = idx % 12;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians12[Hid].m[mid][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 1][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 2][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else if (numbers4 <= idx && idx < numbers3 + numbers4) {
        idx -= numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians9[Hid].m[mid][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 1][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 2][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else if (numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2) {
        idx -= numbers3 + numbers4;
        int Hid = idx / 6;
        int qid = idx % 6;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians6[Hid].m[mid][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 1][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 2][mid + tid];
        gipc::ATOMIC_ADD(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 3;
        int qid = idx % 3;
        gipc::ATOMIC_ADD(&(P[D1Index[Hid]].m[0][qid]), Hessians3[Hid].m[0][qid]);
        gipc::ATOMIC_ADD(&(P[D1Index[Hid]].m[1][qid]), Hessians3[Hid].m[1][qid]);
        gipc::ATOMIC_ADD(&(P[D1Index[Hid]].m[2][qid]), Hessians3[Hid].m[2][qid]);
    }
}

// P = diag(mass) (initialize before accumulating Hessian blocks)
__global__ void __PCG_mass_P(const double* _masses, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double mass = _masses[idx];
    __GEIGEN__::__init_Mat3x3(P[idx], 0);
    P[idx].m[0][0] = mass;
    P[idx].m[1][1] = mass;
    P[idx].m[2][2] = mass;
}

// P = P^{-1}, in place
__global__ void __PCG_inverse_P(__GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    __GEIGEN__::Matrix3x3d PInverse;
    __GEIGEN__::__Inverse(P[idx], PInverse);

    P[idx] = PInverse;
}

// =============================================================================
// 6. Host-side reduction drivers
// =============================================================================

// Collapse the per-block partial sums in squeue down to one scalar on the
// host (second-stage reductions followed by a single DtoH copy).
static double __PCG_reducePartialsToScalar(double* squeue, int numbers) {
    const unsigned int threadNum = default_threads;
    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    int blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(squeue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, squeue, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

// Fused first-stage reduction for the various PCG scalars.
//   type 1: delta0  - energy norm of the filtered RHS
//   type 2: deltaN0 - also initializes r and c
//   type 3: tempSum - q = C*q, then sum q.c
//   type 4: deltaN  - also advances dx/r and evaluates s = P*r
double My_PCG_add_Reduction_Algorithm(int type, device_TetraData* mesh, PCG_Data* pcg_data, int vertexNum, double alpha = 1) {

    int numbers = vertexNum;
    if (numbers < 1)
        return 0;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch (type) {
    case 1:
        PCG_add_Reduction_delta0<<<blockNum, threadNum, sharedMsize>>>(pcg_data->squeue, pcg_data->P, pcg_data->b, mesh->Constraints, numbers);
        break;
    case 2:
        PCG_add_Reduction_deltaN0<<<blockNum, threadNum, sharedMsize>>>(pcg_data->squeue, pcg_data->P, pcg_data->b, pcg_data->r, pcg_data->c, mesh->Constraints, numbers);
        break;
    case 3:
        PCG_add_Reduction_tempSum<<<blockNum, threadNum, sharedMsize>>>(pcg_data->squeue, pcg_data->c, pcg_data->q, mesh->Constraints, numbers);
        break;
    case 4:
        PCG_add_Reduction_deltaN<<<blockNum, threadNum, sharedMsize>>>(pcg_data->squeue, pcg_data->dx, pcg_data->c, pcg_data->r, pcg_data->q, pcg_data->P, pcg_data->s, alpha, numbers);
        break;
    }

    return __PCG_reducePartialsToScalar(pcg_data->squeue, blockNum);
}

// sum_i A[i] . B[i]
double My_PCG_General_v_v_Reduction_Algorithm(device_TetraData* mesh, PCG_Data* pcg_data, double3* A, double3* B, int vertexNum) {

    int numbers = vertexNum;
    if (numbers < 1)
        return 0;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(pcg_data->squeue, A, B, numbers);

    return __PCG_reducePartialsToScalar(pcg_data->squeue, blockNum);
}

// =============================================================================
// 7. Host-side kernel launch wrappers
// =============================================================================

// q = A * c (mass diagonal + all Hessian blocks, fused into two launches)
void Solve_PCG_AX_B2(const device_TetraData* mesh, const double3* c, double3* q, const BHessian& BH, int vertNum) {
    int numbers = vertNum;
    if (numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Solve_AX_mass_b<<<blockNum, threadNum>>>(mesh->masses, c, q, numbers);

    int offset4 = (BH.DNum[3] * 144 + threadNum - 1) / threadNum;
    int offset3 = (BH.DNum[2] * 81 + threadNum - 1) / threadNum;
    int offset2 = (BH.DNum[1] * 36 + threadNum - 1) / threadNum;
    int offset1 = (BH.DNum[0] + threadNum - 1) / threadNum;
    blockNum = offset1 + offset2 + offset3 + offset4;
    __PCG_Solve_AXALL_b2<<<blockNum, threadNum>>>(BH.H12x12, BH.H9x9, BH.H6x6, BH.H3x3, BH.D4Index, BH.D3Index, BH.D2Index, BH.D1Index, c, q, BH.DNum[3] * 144, BH.DNum[2] * 81, BH.DNum[1] * 36, BH.DNum[0], offset4, offset3, offset2);
}

void PCG_Update_Dx_R(const double3* c, double3* dx, const double3* q, double3* r, const double& rate, int vertexNum) {
    int numbers = vertexNum;
    if (numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Update_Dx_R<<<blockNum, threadNum>>>(c, dx, q, r, rate, numbers);
}

// Assemble the block-Jacobi preconditioner: P = (diag blocks of A)^{-1}
void construct_P2(const device_TetraData* mesh, __GEIGEN__::Matrix3x3d* P, const BHessian& BH, int vertNum) {
    int numbers = vertNum;
    if (numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_mass_P<<<blockNum, threadNum>>>(mesh->masses, P, numbers);

    numbers = BH.DNum[3] * 12 + BH.DNum[2] * 9 + BH.DNum[1] * 6 + BH.DNum[0] * 3;
    blockNum = (numbers + threadNum - 1) / threadNum;

    __PCG_AXALL_P<<<blockNum, threadNum>>>(BH.H12x12, BH.H9x9, BH.H6x6, BH.H3x3, BH.D4Index, BH.D3Index, BH.D2Index, BH.D1Index, P, BH.DNum[3] * 12, BH.DNum[2] * 9, BH.DNum[1] * 6, BH.DNum[0] * 3);

    blockNum = (vertNum + threadNum - 1) / threadNum;
    __PCG_inverse_P<<<blockNum, threadNum>>>(P, vertNum);
}

void PCG_FinalStep_UpdateC(const device_TetraData* mesh, double3* c, const double3* s, const double& rate, int vertexNum) {
    int numbers = vertexNum;
    if (numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_FinalStep_UpdateC<<<blockNum, threadNum>>>(mesh->Constraints, s, c, rate, numbers);
}

void PCG_constraintFilter(const device_TetraData* mesh, const double3* input, double3* output, int vertexNum) {
    int numbers = vertexNum;
    if (numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_constraintFilter<<<blockNum, threadNum>>>(mesh->Constraints, input, output, numbers);
}

// =============================================================================
// 8. PCG / MASPCG main loops
// =============================================================================

int MASPCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, int cpNum, double threshold) {
    pcg_data->MP.setPreconditioner(BH, mesh->masses, cpNum);
    double deltaN = 0;
    double delta0 = 0;
    double deltaO = 0;
    CUDA_SAFE_CALL(cudaMemset(pcg_data->dx, 0x0, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(pcg_data->r, 0x0, vertexNum * sizeof(double3)));

    PCG_constraintFilter(mesh, pcg_data->b, pcg_data->filterTempVec3, vertexNum);

    pcg_data->MP.preconditioning(pcg_data->filterTempVec3, pcg_data->preconditionTempVec3);

    delta0 = My_PCG_General_v_v_Reduction_Algorithm(mesh, pcg_data, pcg_data->filterTempVec3, pcg_data->preconditionTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(pcg_data->r, pcg_data->filterTempVec3, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

    PCG_constraintFilter(mesh, pcg_data->preconditionTempVec3, pcg_data->filterTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(pcg_data->c, pcg_data->filterTempVec3, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

    deltaN = My_PCG_General_v_v_Reduction_Algorithm(mesh, pcg_data, pcg_data->r, pcg_data->c, vertexNum);

    double errorRate = threshold;
    int cgCounts = 0;
    while (cgCounts < 3000 && deltaN > errorRate * delta0) {

        cgCounts++;
        Solve_PCG_AX_B2(mesh, pcg_data->c, pcg_data->q, BH, vertexNum);
        double tempSum = My_PCG_add_Reduction_Algorithm(3, mesh, pcg_data, vertexNum);
        double alpha = deltaN / tempSum;
        deltaO = deltaN;
        PCG_Update_Dx_R(pcg_data->c, pcg_data->dx, pcg_data->q, pcg_data->r, alpha, vertexNum);
        pcg_data->MP.preconditioning(pcg_data->r, pcg_data->s);
        deltaN = My_PCG_General_v_v_Reduction_Algorithm(mesh, pcg_data, pcg_data->r, pcg_data->s, vertexNum);
        double rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(mesh, pcg_data->c, pcg_data->s, rate, vertexNum);
    }
    // The solution increment is left in pcg_data->dx.
    return cgCounts;
}



int PCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, double threshold) {
    construct_P2(mesh, pcg_data->P, BH, vertexNum);
    double deltaN = 0;
    double delta0 = 0;
    double deltaO = 0;
    CUDA_SAFE_CALL(cudaMemset(pcg_data->dx, 0x0, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(pcg_data->r, 0x0, vertexNum * sizeof(double3)));
    delta0 = My_PCG_add_Reduction_Algorithm(1, mesh, pcg_data, vertexNum);
    deltaN = My_PCG_add_Reduction_Algorithm(2, mesh, pcg_data, vertexNum);
    double errorRate = threshold;
    int cgCounts = 0;
    while (cgCounts < 30000 && deltaN > errorRate * delta0) {
        cgCounts++;
        Solve_PCG_AX_B2(mesh, pcg_data->c, pcg_data->q, BH, vertexNum);
        double tempSum = My_PCG_add_Reduction_Algorithm(3, mesh, pcg_data, vertexNum);
        double alpha = deltaN / tempSum;
        deltaO = deltaN;
        deltaN = My_PCG_add_Reduction_Algorithm(4, mesh, pcg_data, vertexNum, alpha);
        double rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(mesh, pcg_data->c, pcg_data->s, rate, vertexNum);
    }
    // The solution increment is left in pcg_data->dx.
    return cgCounts;
}

// =============================================================================
// 9. PCG_Data / BHessian device-memory management
// =============================================================================

void PCG_Data::Malloc_DEVICE_MEM(const int& vertexNum, const int& tetrahedraNum) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&squeue, std::max(vertexNum, tetrahedraNum) * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&P, vertexNum * sizeof(__GEIGEN__::Matrix3x3d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&r, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&c, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&z, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&q, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&s, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dx, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(z, 0, vertexNum * sizeof(double3)));

    if (P_type > 0) {
        CUDA_SAFE_CALL(cudaMalloc((void**)&preconditionTempVec3, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&filterTempVec3, vertexNum * sizeof(double3)));
    }
}

void PCG_Data::FREE_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaFree(squeue));
    CUDA_SAFE_CALL(cudaFree(P));
    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(c));
    CUDA_SAFE_CALL(cudaFree(z));
    CUDA_SAFE_CALL(cudaFree(q));
    CUDA_SAFE_CALL(cudaFree(s));
    CUDA_SAFE_CALL(cudaFree(dx));
    if (P_type > 0) {
        CUDA_SAFE_CALL(cudaFree(filterTempVec3));
        CUDA_SAFE_CALL(cudaFree(preconditionTempVec3));
    }
    if (P_type == 1) {
        MP.FreeMAS();
    }
}

void BHessian::updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number) {

    DNum[1] = cpNums[1];
    DNum[2] = cpNums[2] + tri_Num;
    DNum[3] = tet_number + cpNums[3] + tri_edge_number;

#ifdef USE_FRICTION
    DNum[1] += last_cpNums[1];
    DNum[2] += last_cpNums[2];
    DNum[3] += last_cpNums[3];
#endif
}

void BHessian::MALLOC_DEVICE_MEM_O(const int& tet_number,
                                   const int& surfvert_number,
                                   const int& surface_number,
                                   const int& surfEdge_number,
                                   const int& triangle_num,
                                   const int& tri_Edge_number)
{

    int minCollisionBuffer4 = std::max(2 * (surfvert_number + surfEdge_number), 100000);
    int minCollisionBuffer3 = std::max(2 * (surfvert_number + surfEdge_number), 100000);
    int minCollisionBuffer2 = std::max(2 * (surfvert_number + surfEdge_number), 100000);
    int minCollisionBuffer1 = 2 * surfvert_number;


    CUDA_SAFE_CALL(cudaMalloc((void**)&H12x12,
                              (minCollisionBuffer4 + tet_number + tri_Edge_number)
                                  * sizeof(__GEIGEN__::Matrix12x12d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&H9x9,
                              (minCollisionBuffer3 + triangle_num)
                                  * sizeof(__GEIGEN__::Matrix9x9d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&H6x6,
                              minCollisionBuffer2 * sizeof(__GEIGEN__::Matrix6x6d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&H3x3,
                              minCollisionBuffer1 * sizeof(__GEIGEN__::Matrix3x3d)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&D1Index, minCollisionBuffer1 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&D2Index, minCollisionBuffer2 * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&D3Index,
                              (minCollisionBuffer3 + triangle_num) * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&D4Index,
                              (minCollisionBuffer4 + tet_number + tri_Edge_number)
                                  * sizeof(uint4)));
}

void BHessian::FREE_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaFree(H12x12));
    CUDA_SAFE_CALL(cudaFree(H9x9));
    CUDA_SAFE_CALL(cudaFree(H6x6));
    CUDA_SAFE_CALL(cudaFree(H3x3));
    CUDA_SAFE_CALL(cudaFree(D1Index));
    CUDA_SAFE_CALL(cudaFree(D2Index));
    CUDA_SAFE_CALL(cudaFree(D3Index));
    CUDA_SAFE_CALL(cudaFree(D4Index));
}
