#include "runConvection1Dparallel.cuh"

void runConvection1Dparallel()
{
	const int nx = 2000;
	double dx = double(2) / (nx - 1);

	const int nt = 250;
	double dt = 0.001;

	int c = -1;

	double* u, * un;
	cudaMallocManaged(&u, nx * sizeof(double));
	cudaMallocManaged(&un, nx * sizeof(double));

	int threads = 256;
	int blocks = (nx + threads - 1) / (threads);

	// Step function initial condition
	stepFunction1Dparallel << <blocks, threads >> > (dx, nx, u);

	// Finite difference
	finiteDiff1Dparallel << <blocks, threads >> > (c, dt, dx, nt, nx, u, un);

	cudaFree(u);
	cudaFree(un);

	return 0;
}