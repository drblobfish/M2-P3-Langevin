#include <iostream>
#include <fstream>
#include <cstdint>
#define _USE_MATH_DEFINES
#include <cmath>
#include <mutex>
#include <random>
#include <numbers>

double dt = 2e-3;
double epsilon = 1;
double beta = 1;
double _gamma = 1;

uint32_t K = 100000;
const uint32_t N = 100;
const uint32_t nb_epsilon = 10;
const uint32_t nb_gamma = 10;

double integration_res[N*5];

double epsilon_max = 10;
double epsilon_min = 1e-2;
double epsilon_res[nb_epsilon];
double variance_res_epsilon_var[nb_epsilon*5];

double gamma_max = 1e2;
double gamma_min = 1e-4;
double gamma_res[nb_gamma];
double variance_res_gamma_var[nb_gamma*5];

const double rs_mult = 5.2;
const double rs_mu = -0.5;
const double rs_sigma = 1;

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution sample_gaussian {};
std::uniform_real_distribution sample_uniform {};

void geomspace(double a, double b, uint32_t N, double* vals){
        double r = std::exp((std::log(b) - std::log(a))/(N-1));
        vals[0] = a;
        for (uint32_t i = 1; i<N; i++){
                vals[i] = vals[i-1] * r;
        }
}

double U(double q){
        double tmp = q*q - 1;
        return tmp * tmp + 0.5 * q;
}

double grad_U(double q){
        return 4 * q*q*q - 4*q + 0.5;
}

double G_func(double sigma,double zeta,double dt){
        if (zeta == 0){
                return sigma * std::sqrt(dt);
        }else{
                return sigma * std::sqrt((1-std::exp(-2 * dt * zeta))/(2 * zeta));
        }
}

void badodab_step(double *p, double *q, double *xi){
        double p_half = *p - 0.5 * dt * grad_U(*q);
        double q_half = *q + 0.5 * dt *p_half;
        double xi_half = *xi + 0.5 * dt * (p_half * p_half - 1/beta)/epsilon;
        double alpha = std::exp(- dt * (xi_half/epsilon + _gamma));
        double G = G_func(std::sqrt(2 * _gamma / beta),xi_half/epsilon + _gamma,dt);
        double R = sample_gaussian(gen);
        double p_hat = alpha * p_half + G * R;
        *xi = xi_half + 0.5 * dt * (p_hat*p_hat - 1/beta)/epsilon;
        *q = q_half + 0.5 * dt * p_hat;
        *p = p_hat - 0.5 * dt * grad_U(*q);
}

void badodab_integrate_all(double p,double q,double xi, double *integral){
        for (uint32_t i = 0; i<K; i++){
                badodab_step(&p,&q,&xi);
                integral[0] += p;
                integral[1] += q;
                integral[2] += xi;
                integral[3] += q*q;
                integral[4] += xi*xi;
        }
        integral[0] /= K;
        integral[1] /= K;
        integral[2] /= K;
        integral[3] /= K;
        integral[4] /= K;
}

void sample_invariant_measure_rejection(double *p0,double *q0,double *xi0){
        *p0 = sample_gaussian(gen);
        *xi0 = sample_gaussian(gen);
        while (true){
                double q = sample_gaussian(gen) * rs_sigma - rs_mu;
                double u = sample_uniform(gen);
                double y = rs_mult * 1/(std::sqrt(2*M_PI) * rs_sigma)
                        * std::exp(- (q - rs_mu)*(q - rs_mu) /(2 * rs_sigma*rs_sigma));
                if (y * u < std::exp(-U(q))){
                        *q0 = q;
                        return;
                }
        }
}

void badodab_variance(double *variances){
        double means[5] = {0};
        for (uint32_t j = 0; j<5; j++){
                variances[j] = 0;
        }
        for (uint32_t i = 0; i<N; i++){
                double p0;
                double q0;
                double xi0;
                sample_invariant_measure_rejection(&p0,&q0,&xi0);
                badodab_integrate_all(p0,q0,xi0,integration_res+i*5);
                for (uint32_t j = 0; j<5; j++){
                        means[j] += (integration_res+i*5)[j];
                }
        }
        for (uint32_t j = 0; j<5; j++){
                means[j] /= N;
        }
        for (uint32_t i = 0; i<N; i++){
                for(uint32_t j=0; j<5; j++){
                        double diff = integration_res[i*5+j] - means[j];
                        variances[j] += diff * diff;
                }
        }
        for (uint32_t j = 0; j<5; j++){
                variances[j] /= N;
        }
}

void dump_results(){
        std::ofstream f1("exp1_epsilon.csv");
        f1 << "epsilon,p,q,xi,q^2,xi^2\n";
        for (uint32_t i = 0; i<nb_epsilon;i++){
                f1 << epsilon_res[i];
                for (uint32_t j = 0; j<5;j++){
                        f1 << "," << variance_res_epsilon_var[i*5 + j];
                }
                f1 << "\n";
        }
        std::ofstream f2("exp2_gamma.csv");
        f2 << "gamma,p,q,xi,q^2,xi^2\n";
        for (uint32_t i = 0; i<nb_gamma;i++){
                f2 << gamma_res[i];
                for (uint32_t j = 0; j<5;j++){
                        f2 << "," << variance_res_gamma_var[i*5 + j];
                }
                f2 << "\n";
        }
}

int main(){
        geomspace(epsilon_min,epsilon_max,nb_epsilon,epsilon_res);
        geomspace(gamma_min,gamma_max,nb_gamma,gamma_res);
        // experiment 1
        _gamma = 1;
        for (uint32_t i = 0; i<nb_epsilon;i++){
                std::cout << "Experiment 1 ( gamma = " << _gamma << ", epsilon = " << epsilon << ") \n";
                epsilon = epsilon_res[i];
                badodab_variance(variance_res_epsilon_var+i*5);
        }
        // experiment 2
        epsilon = 1;
        for (uint32_t i = 0; i<nb_gamma;i++){
                std::cout << "Experiment 2 ( gamma = " << _gamma << ", epsilon = " << epsilon << ") \n";
                _gamma = gamma_res[i];
                badodab_variance(variance_res_gamma_var+i*5);
        }
        dump_results();
        return 0;
}
