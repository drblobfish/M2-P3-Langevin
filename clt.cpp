#include <iostream>
#include <fstream>
#include <cstdint>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <numbers>

double dt = 1e-1;
double epsilon = 1;
double beta = 1;
double _gamma = 1;
std::vector<double> Ts = {5.0,7.5,10.0,50.0};
const uint32_t N = 50000;
uint32_t K = 0;
uint32_t K_max = 1000;

double integration_res[N*5] = {0};

const double rs_mult = 5.2;
const double rs_mu = -0.5;
const double rs_sigma = 1;

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution sample_gaussian {};
std::uniform_real_distribution sample_uniform {};

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

void badodab_variance_mean(double *means, double *variances){
        for (uint32_t j = 0; j<5; j++){
                means[j] = 0;
                variances[j] = 0;
        }
        for (uint32_t i = 0; i<N; i++){
                if (i % 1000 == 0){
                        std::cout << "variance estimation ("<< 100 * i / N<<"%)\n";
                }
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

int main(){
        double var[5] = {0};
        double means[5] = {0};
        K = K_max;
        badodab_variance_mean(means,var);
        std::ofstream f_var("exp_clt_var.txt");
        f_var << var[1] << "\n" << var[3] << "\n";

        std::ofstream f_clt("exp_clt.csv");
        f_clt << "T,var,err\n";
        for (double T : Ts){
                std::cout << "exp T="<<T<<"\n";
                K = (uint32_t) T/dt;
                for (uint32_t i = 0; i<N; i++){
                        double p0;
                        double q0;
                        double xi0;
                        double integral[5] = {0};
                        sample_invariant_measure_rejection(&p0,&q0,&xi0);
                        badodab_integrate_all(p0,q0,xi0,integral);
                        f_clt << T << ",q," << std::sqrt(T/var[1])*(integral[1]-means[1]) << "\n";
                        f_clt << T << ",q²," << std::sqrt(T/var[3])*(integral[3]-means[3]) << "\n";
                }
        }
        return 0;
}
