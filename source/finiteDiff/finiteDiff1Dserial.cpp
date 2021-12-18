#include "finiteDiff1Dserial.hpp"

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