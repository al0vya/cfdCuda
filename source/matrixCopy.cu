#include "matrixCopy.cuh"

__global__
void matrixCopy
(
	double d_A[][],
	double d_B[][],
	const int N
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N && iy < N) {
		d_B[ix][iy] = d_A[ix][iy];
	}
}