#include "stepFunction1Dparallel.cuh"

__global__
void stepFunction1Dparallel(const double dx, const int nx, double* u) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < nx; i += stride) {

		if (i * dx >= 0.5 && i * dx <= 1) {
			u[i] = 2;
		}
		else {
			u[i] = 1;
		}
	}
}