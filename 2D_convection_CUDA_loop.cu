#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

// MAX THREADS PER BLOCK FOR GTX 1050 MOBILE GPU CARD IS 256
#define N 32


__global__ void matrixCopy(double d_A[N][N], double d_B[N][N]) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N && iy < N) {
		d_B[ix][iy] = d_A[ix][iy];
	}
}

__global__ void stepFunction2D(double d_A[N][N], double d) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	for (iy; iy < N; iy++) {
		for (ix; ix < N; ix++) {

			if (ix * d >= 0.5 && ix * d <= 1 && iy * d >= 0.5 && iy * d <= 1) {
				d_A[ix][iy] = 2;
			}
			else {
				d_A[ix][iy] = 1;
			}
		}
	}
}

__global__ void finiteDiff2D(double d_A[N][N], double d_B[N][N], const double d, const double dt, const double c) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	for (iy; iy < N; iy++) {
		for (ix; ix < N; ix++) {

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
	}
}

int main()
{
	// Declaring on host
	double A[N][N];
	double B[N][N];

	double d = double(2) / N;
	int nt = 10;
	double c = 0.2;
	double sigma = 0.2;
	double dt = sigma * d;

	cudaError_t cudaStatus;

	// Device pointers
	double(*d_A)[N], (*d_B)[N];

	// Allocate host memory
	cudaStatus = cudaMalloc((void**)& d_A, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)& d_B, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Copy to device from host
	cudaStatus = cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// One block only
	int blocks = 1;

	// One thread per element
	dim3 threads(N, N);

	stepFunction2D <<<blocks, threads >>> (d_A, d);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	for (int t = 0; t < nt; t++) {
		matrixCopy<<<blocks, threads>>>(d_A, d_B);
		finiteDiff2D<<<blocks, threads>>>(d_A, d_B, d, dt, c);
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	}

	// Copy back to host from device
	cudaStatus = cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << (A[i][j]) << " ";
		}
		std::cout << endl;
	}

	std::cout << endl;

	cudaFree(d_A);
	cudaFree(d_B);
}