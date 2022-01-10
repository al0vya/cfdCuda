#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <ctime>

#define N 32

void finiteDiff1Dserial
(
	const int& c,
	const double& dt,
	const double& dx,
	const int& nt,
	const int& nx,
	double* u,
	double* un
);

__global__
void stepFunction1Dparallel
(
    const double dx, 
    const int nx, 
    double* u
);

__global__
void finiteDiff1Dparallel
(
	const int c,
	const double dt,
	const double dx,
	const int nt,
	const int nx,
	double* u,
	double* un
);

__global__
void stepFunction2Dparallel
(
	double d_A[N][N],
	double d
);

__global__
void finiteDiff2Dparallel
(
	double d_A[N][N],
	double d_B[N][N],
	const double d,
	const double dt,
	const double c
);

void runConvection1Dserial();

void runConvection1Dparallel();

void runConvection2Dparallel();

__global__
void matrixCopy
(
	double d_A[N][N],
	double d_B[N][N]
);

int main()
{
	runConvection1Dserial();
	runConvection1Dparallel();
	runConvection2Dparallel();

	return 0;
}

void finiteDiff1Dserial
(
	const int& c, 
	const double& dt, 
	const double& dx, 
	const int& nt, 
	const int& nx, 
	double* u, 
	double* un
)
{
	for (int t = 0; t < nt; t++)
	{
		for (int i = 1; i < nx; i++)
		{
			un[i] = u[i];
			u[i]  = un[i] - c * dt / dx * ( un[i] - un[i - 1] );
		}
	}
}

__global__
void finiteDiff1Dparallel
(
	const int c,
	const double dt,
	const double dx,
	const int nt,
	const int nx,
	double* u,
	double* un
)
{

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

__global__
void stepFunction2Dparallel
(
	double d_A[N][N],
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

__global__
void finiteDiff2Dparallel
(
	double d_A[N][N],
	double d_B[N][N],
	const double d,
	const double dt,
	const double c
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

void runConvection1Dserial()
{
	const int nx = 2000;
	double dx = double(2) / (nx - 1);

	const int nt = 250;
	double dt = 0.001;

	int c = 1;

	double* u = new double[nx];
	double* un = new double[nx];

	clock_t start = clock();

	// Step function initial condition
	for (int i = 0; i < nx; i++) {

		if (i * dx >= 0.5 && i * dx <= 1) {
			u[i] = 2;
		}
		else {
			u[i] = 1;
		}
	}

	finiteDiff1Dserial(c, dt, dx, nt, nx, u, un);

	clock_t end = clock();

	double runtime = (start - end) / (double)CLOCKS_PER_SEC;

	printf("\nSerial 1D runtime: %f ms\n", runtime);

	delete[] u;
	delete[] un;
}

void runConvection1Dparallel()
{
	const int nx = 2000;
	double dx = double(2) / (nx - 1);

	const int nt = 250;
	double dt = 0.001;

	int c = 1;

	double* u, * un;
	cudaMallocManaged(&u, nx * sizeof(double));
	cudaMallocManaged(&un, nx * sizeof(double));

	int threads = 256;
	int blocks = (nx + threads - 1) / (threads);

	cudaEvent_t cuda_begin, cuda_end;
	cudaEventCreate(&cuda_begin);
	cudaEventCreate(&cuda_end);

	cudaEventRecord(cuda_begin);

	// Step function initial condition
	stepFunction1Dparallel<<<blocks, threads>>>(dx, nx, u);

	// Finite difference
	finiteDiff1Dparallel<<<blocks, threads>>>(c, dt, dx, nt, nx, u, un);
	
	cudaEventRecord(cuda_end);
	cudaEventSynchronize(cuda_end);

	float cuda_time = 0;

	cudaEventElapsedTime(&cuda_time, cuda_begin, cuda_end);

	printf("\nParallel 1D runtime: %f ms\n", cuda_time);
	
	cudaFree(u);
	cudaFree(un);
}

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
	
	cudaEvent_t cuda_begin, cuda_end;
	cudaEventCreate(&cuda_begin);
	cudaEventCreate(&cuda_end);

	cudaEventRecord(cuda_begin);

	stepFunction2Dparallel<<<blocks, threads>>>(d_A, d);

	for (int t = 0; t < nt; t++) {
		matrixCopy<<<blocks, threads>>>(d_A, d_B);
		finiteDiff2Dparallel<<<blocks, threads>>>(d_A, d_B, d, dt, c);
	}
	
	cudaEventRecord(cuda_end);
	cudaEventSynchronize(cuda_end);

	float cuda_time = 0;

	cudaEventElapsedTime(&cuda_time, cuda_begin, cuda_end);

	printf("\nParallel 2D runtime: %f ms\n", cuda_time);
	
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
}

__global__
void matrixCopy
(
	double d_A[N][N],
	double d_B[N][N]
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < N && iy < N) {
		d_B[ix][iy] = d_A[ix][iy];
	}
}