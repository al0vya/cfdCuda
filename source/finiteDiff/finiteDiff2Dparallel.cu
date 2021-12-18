#include "finiteDiff2Dparallel.cuh"

__global__
void finiteDiff2Dparallel
(
	double d_A[][],
	double d_B[][],
	const double d,
	const double dt,
	const double c,
	const int N
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N - 1 && iy < N - 1) {
		d_A[ix + 1][iy + 1] = d_B[ix + 1][iy + 1]
			- c * dt / d * (d_B[ix + 1][iy + 1] - d_B[ix + 1][iy])
			- c * dt / d * (d_B[ix + 1][iy + 1] - d_B[ix][iy + 1]);
	}

	// Boundary conditions
	if (ix == 0 || ix == N || iy == 0 || iy == N) {
		d_A[ix][iy] = 1;
	}
}