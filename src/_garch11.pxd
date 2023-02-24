cdef extern from "garch11.hpp" nogil:
    double calc_backcast(double *r2, int n)
    double calc_fun(double *x, double *r2, int n)
    void calc_jac(double *x, double *r2, int n, double *out_jac)
    double calc_fun_jac(double *x, double *r2, int n, double *out_jac)
    double calc_fun_bc(double *x, double *r2, int n)
    void calc_jac_bc(double *x, double *r2, int n, double *out_jac)
    double calc_fun_jac_bc(double *x, double *r2, int n, double *out_jac)
    void transform(double *x, double *r2, int n, double *out_sigma2)
    void transform_bc(double *x, double *r2, int n, double *out_sigma2)
    void predict(double *x, double last_sigma2, double last_r2,
                 double *randn_nums, int n, double *out_sigma2, double *out_r2)
    double nll(double *r2, int n, double *sigma2)
