#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void stepFunction2Dparallel
(
	double d_A[][],
	double d
);