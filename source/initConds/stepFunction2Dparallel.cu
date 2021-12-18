#include "stepFunction2Dparallel.cuh"

__global__
void stepFunction2Dparallel
(
	double d_A[][],
	double d
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix * d >= 0.5 && ix * d <= 1 && iy * d >= 0.5 && iy * d <= 1) {
		d_A[ix][iy] = 2;
	}
	else {
		d_A[ix][iy] = 1;
	}
}