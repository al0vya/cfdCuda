#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void finiteDiff2Dparallel
(
	double d_A[][],
	double d_B[][],
	const double d,
	const double dt,
	const double c,
	const int N
);