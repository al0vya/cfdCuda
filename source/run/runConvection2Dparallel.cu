#pragma once

#include "runConvection2Dparallel.cuh"

#define N 32

void runConvection2Dparallel()
{
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
	double(*d_A)[N], (*d_B)[N], (*d_C)[N];

	// Allocate host memory
	cudaStatus = cudaMalloc((void**)&d_A, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_B, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_C, N * N * sizeof(double));
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

	stepFunction2D << <blocks, threads >> > (d_A, d);

	for (int t = 0; t < nt; t++) {
		matrixCopy << <blocks, threads >> > (d_A, d_B);
		finiteDiff2D << <blocks, threads >> > (d_A, d_B, d, dt, c);
	}

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

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}