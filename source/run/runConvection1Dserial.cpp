#include "runConvection1Dserial.hpp"

void runConvection1Dserial()
{
	const int nx = 2000;
	double dx = double(2) / (nx - 1);

	const int nt = 25;
	double dt = 0.001;

	int c = 1;

	double* u = new double[nx];
	double* un = new double[nx];

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

	delete[] u;
	delete[] un;
}