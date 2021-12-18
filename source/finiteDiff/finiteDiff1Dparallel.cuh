#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void finiteDiff1Dparallel
(
	const int c,
	const double dt,
	const double dx,
	const int nt,
	const int nx,
	double* u,
	double* un
);