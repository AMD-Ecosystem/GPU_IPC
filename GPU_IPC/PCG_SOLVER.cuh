//
// PCG_SOLVER.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//
// GPU preconditioned conjugate gradient solver for the IPC Newton system.
//
//   PCG_Data        - device workspace (vectors, preconditioner, scalars)
//   PCG_Process     - classic PCG with the block-Jacobi (3x3) preconditioner
//   MASPCG_Process  - PCG with the MAS preconditioner
//

#pragma once
#ifndef _PCG_SOLVER_CUH_
#define _PCG_SOLVER_CUH_
#include <cuda_runtime.h>
#include "device_fem_data.cuh"
#include <cstdint>
#include "MASPreconditioner.cuh"

class PCG_Data {
public:
	double* squeue;                 // partial-sum queue for reductions
	double3* b;                     // right-hand side (owned by caller)
	__GEIGEN__::Matrix3x3d* P;      // block-Jacobi preconditioner blocks
	double3* r;                     // residual
	double3* c;                     // search direction
	double3* q;                     // A * c
	double3* s;                     // preconditioned residual
	double3* z;                     // initial guess workspace
	double3* dx;                    // solution increment

	double3* filterTempVec3;        // scratch for the constraint filter (MAS)
	double3* preconditionTempVec3;  // scratch for preconditioner output (MAS)
	MASPreconditioner MP;

	int P_type;

public:
	void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num);
	void FREE_DEVICE_MEM();
};

int PCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, double threshold);
int MASPCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, int cpNum, double threshold);
#endif // ! _PCG_SOLVER_CUH_
