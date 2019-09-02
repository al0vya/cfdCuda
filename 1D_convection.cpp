#include <iostream>
#include <cmath>

using namespace std;

void finiteDiff(const int& c, const double& dt, const double& dx, const int& nt, const int& nx, double* u, double* un) {
	for (int t = 0; t < nt; t++) {

		for (int i = 1; i < nx; i++) {
			un[i] = u[i];
			u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1]);
		}
	}
}

int main() {

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

		cout << u[i] << endl;
	}

	cout << endl;

	/* // Finite difference loop
	for (int t = 0; t < nt; t++) {

		for (int i = 0; i < nx; i++) {
			un[i] = u[i];
		}

		for (int i = 1; i < nx; i++) {
			u[i] = un[i] - c * dt / dx * (un[i] - un[i-1]);
		}
	}
	*/

	finiteDiff(c, dt, dx, nt, nx, u, un);

	for (int i = 0; i < nx; i++) {
		cout << u[i] << endl;
	}

	cout << "Solved";

	delete[] u;
	delete[] un;

	return 0;
}