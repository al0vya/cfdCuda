#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

// MAX THREADS PER BLOCK FOR GTX 1050 MOBILE GPU CARD IS 256
#define N 32

__global__ void matrixAdd(double d_A[][N], double d_B[][N], double d_C[][N]) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N && iy < N) {
		d_C[ix][iy] = d_A[ix][iy] + d_B[ix][iy];
	}
}

__global__ void matrixDefine(double d_A[N][N]) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N && iy < N) {
		d_A[ix][iy] = 1;
	}
}

__global__ void stepFunction2D(double d_A[N][N], double d) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix * d >= 0.5 && ix * d <= 1 && iy * d >= 0.5 && iy * d <= 1) {
		d_A[ix][iy] = 2;
	}
	else {
		d_A[ix][iy] = 1;
	}
}

__global__ void matrixCopy(double d_A[N][N], double d_B[N][N]) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N && iy < N) {
		d_B[ix][iy] = d_A[ix][iy];
	}
}

__global__ void finiteDiff2D(double d_A[N][N], double d_B[N][N], const double d, const double dt, const double c) {
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

int main() {

	// Declaring on host
	double A[N][N];
	double B[N][N];
	double C[N][N];
	
	double d = double(2) / N;
	int nt = 10;
	double c = 0.2;
	double sigma = 0.2;
	double dt = sigma * d;

	cudaError_t cudaStatus;

	// Device pointers
	double (*d_A)[N], (*d_B)[N], (*d_C)[N];

	// Allocate host memory
	cudaStatus = cudaMalloc((void**)& d_A, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)& d_B, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	
	cudaStatus = cudaMalloc((void**)& d_C, N * N * sizeof(double));
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

	cudaStatus = cudaMemcpy(d_C, C, N * N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// One block only
	int blocks = 1;

	// One thread per element
	dim3 threads(N, N);

	stepFunction2D<<<blocks, threads>>>(d_A, d);
	
	for (int t = 0; t < nt; t++) {
		matrixCopy<<<blocks, threads>>>(d_A, d_B);
		finiteDiff2D<<<blocks, threads>>>(d_A, d_B, d, dt, c);
	}
	
//	stepFunction2D << <blocks, threads >> > (d_A, d);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	}

	// Copy back to host from device
	cudaStatus = cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	
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

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}