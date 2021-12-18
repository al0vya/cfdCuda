#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void matrixCopy
(
	double d_A[][],
	double d_B[][],
	const int N
);