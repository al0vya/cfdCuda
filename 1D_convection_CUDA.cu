
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;


__global__ void finiteDiff(const int c, const double dt, const double dx, const int nt, const int nx, double *u, double *un) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int t = 0; t < nt; t++) {

		for (int i = index; i < nx; i += stride) {
			un[i] = u[i];
		}

		for (int i = index + 1; i < nx; i += stride) {
			u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1]);
		}
	}
}

__global__ void stepFunction(const double dx, const int nx, double *u) {
	
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

int main()
{
	const int nx = 2000;
	double dx = double(2) / (nx - 1);

	const int nt = 250;
	double dt = 0.001;

	int c = -1;

	double *u, *un;
	cudaMallocManaged(&u, nx * sizeof(double));
	cudaMallocManaged(&un, nx * sizeof(double));

	int threads = 256;
	int blocks = (nx + threads - 1) / (threads);

	// Step function initial condition
	stepFunction<<<blocks, threads >>>(dx, nx, u);

	cudaDeviceSynchronize();

	// Visualise step function
	ofstream myfile("data.txt");

	for (int i = 0; i < nx; i++) {
		myfile << u[i] << " ";
	}

	myfile << endl << endl << endl;

	// Finite difference
	finiteDiff<<<blocks, threads>>>(c, dt, dx, nt, nx, u, un);

	cudaDeviceSynchronize();

	// Visualise final solution
	for (int i = 0; i < nx; i++) {
		myfile << u[i] << " ";
	}

	cout << "Solved";

	cudaFree(u);
	cudaFree(un);

	return 0;
}