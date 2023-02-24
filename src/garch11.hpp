#ifndef _GARCH11_H_
#define _GARCH11_H_

#include <cmath>

/// @param r2 square of log returns
/// @param n the length of r2
/// @return the backcast initialization
double calc_backcast(double *, int);

/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @return the negative loglikelihood
double calc_fun(double *, double *, int);

/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @param out_jac to which to write partial derivatives of Garch11 parameters,
///        which should be of length 3
void calc_jac(double *, double *, int, double *);

/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @param out_jac to which to write partial derivatives of Garch11 parameters,
///        which should be of length 3
/// @return the negative loglikelihood
double calc_fun_jac(double *, double *, int, double *);

/// Same as calc_fun but using backcast as initialization.
/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @return the negative loglikelihood
double calc_fun_bc(double *, double *, int);

/// Same as calc_jac but using backcast as initialization.
/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @param out_jac to which to write partial derivatives of Garch11 parameters,
///        which should be of length 3
void calc_jac_bc(double *, double *, int, double *);

/// Same as calc_jac but using backcast as initialization.
/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @param out_jac to which to write partial derivatives of Garch11 parameters,
///        which should be of length 3
/// @return the negative loglikelihood
double calc_fun_jac_bc(double *, double *, int, double *);

/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @param out_sigma2 to which to write transformation result, which should be
///        of length n
void transform(double *, double *, int, double *);

/// Same as transform but using backcast as initialization.
/// @param x the Garch11 parameters, which should be of length 3
/// @param r2 square of log returns
/// @param n the length of r2
/// @param out_sigma2 to which to write transformation result, which should be
///        of length n
void transform_bc(double *, double *, int, double *);

/// @param x the Garch11 parameters, which should be of length 3
/// @param last_sigma2 last sigma2
/// @param last_r2 square of the last log return
/// @param randn_nums an array of standard normal random variables
/// @param n the length of randn_nums
/// @param out_sigma2 to which to write prediction result, which should be of
///        length n
/// @param out_r2 to which to write prediction result, which should be of length
///        n
void predict(double *, double, double, double *, int, double *, double *);

/// @param r2 square of log returns
/// @param n the length of r2
/// @param sigma2 the conditional volatilities, which should be of length n
/// @return the negative log likelihood
double nll(double *, int, double *);

#endif
