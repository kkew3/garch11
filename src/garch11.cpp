#include "garch11.hpp"

using std::log;
using std::sqrt;

double calc_backcast(double *r2, int n) {
    int tau = 75 < n ? 75 : n;
    double w;
    double sum = 0.0;
    for (int i = 0; i < tau; ++i) {
        w = static_cast<double>(i) / (0.5 * tau * (tau - 1));
        sum += r2[i] * w;
    }
    return sum;
}

double calc_fun(double *x, double *r2, int n) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double sigma2 = omega / (1 - alpha - beta);
    double y = log(sigma2) + r2[i] / sigma2;
    for (i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2;
        y += log(sigma2) + r2[i] / sigma2;
    }
    return y;
}

void calc_jac(double *x, double *r2, int n, double *out_jac) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double sigma2 = omega / (1.0 - alpha - beta);

    double par_sigma2_omega = 1.0 / (1.0 - alpha - beta);
    double par_sigma2_alpha = omega * par_sigma2_omega * par_sigma2_omega;
    double par_sigma2_beta = par_sigma2_alpha;

    double d_y_sigma2 = 1.0 / sigma2 * (1.0 - r2[i] / sigma2);
    double par_y_omega = d_y_sigma2 * par_sigma2_omega;
    double par_y_alpha = d_y_sigma2 * par_sigma2_alpha;
    double par_y_beta = d_y_sigma2 * par_sigma2_beta;
    double sigma2_prev = sigma2;
    for (i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        par_sigma2_omega = 1.0 + beta * par_sigma2_omega;
        par_sigma2_alpha = r2[i - 1] + beta * par_sigma2_alpha;
        par_sigma2_beta = sigma2_prev + beta * par_sigma2_beta;
        d_y_sigma2 = 1.0 / sigma2 * (1.0 - r2[i] / sigma2);
        par_y_omega += d_y_sigma2 * par_sigma2_omega;
        par_y_alpha += d_y_sigma2 * par_sigma2_alpha;
        par_y_beta += d_y_sigma2 * par_sigma2_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
}

double calc_fun_jac(double *x, double *r2, int n, double *out_jac) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double sigma2 = omega / (1 - alpha - beta);
    double y = log(sigma2) + r2[i] / sigma2;

    double par_sigma2_omega = 1 / (1 - alpha - beta);
    double par_sigma2_alpha = omega * par_sigma2_omega * par_sigma2_omega;
    double par_sigma2_beta = par_sigma2_alpha;

    double d_y_sigma2 = 1 / sigma2 * (1 - r2[i] / sigma2);
    double par_y_omega = d_y_sigma2 * par_sigma2_omega;
    double par_y_alpha = d_y_sigma2 * par_sigma2_alpha;
    double par_y_beta = d_y_sigma2 * par_sigma2_beta;
    double sigma2_prev = sigma2;
    for (i = 1; i < n; ++i)
    {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        y += log(sigma2) + r2[i] / sigma2;
        par_sigma2_omega = 1 + beta * par_sigma2_omega;
        par_sigma2_alpha = r2[i - 1] + beta * par_sigma2_alpha;
        par_sigma2_beta = sigma2_prev + beta * par_sigma2_beta;
        d_y_sigma2 = 1 / sigma2 * (1 - r2[i] / sigma2);
        par_y_omega += d_y_sigma2 * par_sigma2_omega;
        par_y_alpha += d_y_sigma2 * par_sigma2_alpha;
        par_y_beta += d_y_sigma2 * par_sigma2_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
    return y;
}

double calc_fun_bc(double *x, double *r2, int n) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double bc = calc_backcast(r2, n);
    double sigma2 = omega + (alpha + beta) * bc;
    double y = log(sigma2) + r2[i] / sigma2;
    for (i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2;
        y += log(sigma2) + r2[i] / sigma2;
    }
    return y;
}

void calc_jac_bc(double *x, double *r2, int n, double *out_jac) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double bc = calc_backcast(r2, n);
    double sigma2 = omega + (alpha + beta) * bc;

    double par_sigma2_omega = 1.0;
    double par_sigma2_alpha = bc;
    double par_sigma2_beta = bc;

    double d_y_sigma2 = 1.0 / sigma2 * (1.0 - r2[i] / sigma2);
    double par_y_omega = d_y_sigma2 * par_sigma2_omega;
    double par_y_alpha = d_y_sigma2 * par_sigma2_alpha;
    double par_y_beta = d_y_sigma2 * par_sigma2_beta;
    double sigma2_prev = sigma2;
    for (i = 1; i < n; ++i) {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        par_sigma2_omega = 1.0 + beta * par_sigma2_omega;
        par_sigma2_alpha = r2[i - 1] + beta * par_sigma2_alpha;
        par_sigma2_beta = sigma2_prev + beta * par_sigma2_beta;
        d_y_sigma2 = 1.0 / sigma2 * (1.0 - r2[i] / sigma2);
        par_y_omega += d_y_sigma2 * par_sigma2_omega;
        par_y_alpha += d_y_sigma2 * par_sigma2_alpha;
        par_y_beta += d_y_sigma2 * par_sigma2_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
}

double calc_fun_jac_bc(double *x, double *r2, int n, double *out_jac) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double bc = calc_backcast(r2, n);
    double sigma2 = omega + (alpha + beta) * bc;
    double y = std::log(sigma2) + r2[i] / sigma2;

    double par_sigma2_omega = 1.0;
    double par_sigma2_alpha = bc;
    double par_sigma2_beta = bc;

    double d_y_sigma2 = 1 / sigma2 * (1 - r2[i] / sigma2);
    double par_y_omega = d_y_sigma2 * par_sigma2_omega;
    double par_y_alpha = d_y_sigma2 * par_sigma2_alpha;
    double par_y_beta = d_y_sigma2 * par_sigma2_beta;
    double sigma2_prev = sigma2;
    for (i = 1; i < n; ++i)
    {
        sigma2 = omega + alpha * r2[i - 1] + beta * sigma2_prev;
        y += std::log(sigma2) + r2[i] / sigma2;
        par_sigma2_omega = 1 + beta * par_sigma2_omega;
        par_sigma2_alpha = r2[i - 1] + beta * par_sigma2_alpha;
        par_sigma2_beta = sigma2_prev + beta * par_sigma2_beta;
        d_y_sigma2 = 1 / sigma2 * (1 - r2[i] / sigma2);
        par_y_omega += d_y_sigma2 * par_sigma2_omega;
        par_y_alpha += d_y_sigma2 * par_sigma2_alpha;
        par_y_beta += d_y_sigma2 * par_sigma2_beta;
        sigma2_prev = sigma2;
    }
    out_jac[0] = par_y_omega;
    out_jac[1] = par_y_alpha;
    out_jac[2] = par_y_beta;
    return y;
}

void transform(double *x, double *r2, int n, double *out_sigma2) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    out_sigma2[i] = omega / (1.0 - alpha - beta);
    for (i = 1; i < n; ++i) {
        out_sigma2[i] = omega + alpha * r2[i - 1] + beta * out_sigma2[i - 1];
    }
}

void transform_bc(double *x, double *r2, int n, double *out_sigma2) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    double bc = calc_backcast(r2, n);
    out_sigma2[i] = omega + (alpha + beta) * bc;
    for (i = 1; i < n; ++i) {
        out_sigma2[i] = omega + alpha * r2[i - 1] + beta * out_sigma2[i - 1];
    }
}

void predict(double *x, double last_sigma2, double last_r2, double *randn_nums,
        int n, double *out_sigma2, double *out_r2) {
    double omega = x[0];
    double alpha = x[1];
    double beta = x[2];

    int i = 0;
    out_sigma2[i] = omega + alpha * last_r2 + beta * last_sigma2;
    out_r2[i] = randn_nums[i] * sqrt(out_sigma2[i]);
    for (i = 1; i < n; ++i) {
        out_sigma2[i] = omega + alpha * out_r2[i - 1] * out_r2[i - 1]
            + beta * out_sigma2[i - 1];
        out_r2[i] = randn_nums[i] * sqrt(out_sigma2[i]);
    }
}

double nll(double *r2, int n, double *sigma2) {
    double y = 0.0;
    for (int i = 0; i < n; ++i) {
        y += log(sigma2[i]) + r2[i] / sigma2[i];
    }
    return y;
}
