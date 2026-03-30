import numpy as np
import matplotlib.pyplot as plt

def badodab_step(p,q,xi,grad_eval,dt,epsilon,beta,gamma):
    def G_func(sigma,zeta,dt):
        if zeta == 0:
            return sigma * np.sqrt(dt)
        else :
            return sigma * np.sqrt((1-np.exp(-2 * dt * zeta))/(2 * zeta))

    p_half = p - 0.5 * dt * grad_eval(q)
    q_half = q + 0.5 * dt * p_half
    xi_half = xi + 0.5 * dt * (np.sum(p_half**2) - (p.shape[0]/beta))/epsilon
    alpha = np.exp(-dt * (xi_half/epsilon + gamma))
    G = G_func(np.sqrt(2 * gamma / beta),xi_half/epsilon + gamma,dt)
    R = np.random.normal(0,1,p.shape)
    p_hat = alpha * p_half + G * R
    xi_new = xi_half + 0.5 * dt * (np.sum(p_hat**2) - (p.shape[0]/beta))/epsilon
    q_new = q_half + 0.5 * dt * p_hat
    p_new = p_hat - 0.5 * dt * grad_eval(q_new)
    return (p_new,q_new,xi_new)

def sample_invariant_measure_rejection(mult,mu,sigma,U):
    Z = np.sum(np.exp(-U(np.linspace(-2,2,1000))) * 4/1000)
    p = np.random.normal(0,1)
    xi = np.random.normal(0,1)
    while True :
        q = np.random.normal(mu,sigma)
        u = np.random.uniform()
        y = mult * 1/(np.sqrt(2 * np.pi)*sigma) * np.exp(-(q-mu)**2 /(2*sigma**2))
        if y*u < np.exp(-U(q))/Z:
            return (p,q,xi)


# validation rejection sampling
mu = -0.4
sigma = 1
mult = 2.5

qs = np.empty(100000)
for i in range(100000):
    p,q,xi = sample_invariant_measure_rejection(mult,mu,sigma,U)
    qs[i] = q

plt.hist(qs,bins=100,density=True)
plt.plot(np.linspace(-2,2,100),np.exp(-U(np.linspace(-2,2,100)))/Z,"r")
plt.show()

#

K = 100_000
dt = 2e-2
epsilon = 1
beta = 10
gamma = 1

def U(q):
    return (q**2 - 1)**2 + 0.5 * q

def gradU(q):
    return 4 * q**3 - 4*q + 0.5

p0 = np.array([0])
q0 = np.array([0])
xi0 = 1

every = 100
q_vec = np.empty(K//every)
for i in range(K):
    (p0,q0,xi0) = badodab_step(p0,q0,xi0,gradU,dt,epsilon,beta,gamma)
    if i%every == 0:
        q_vec[i//every] = q0[0]

x = np.linspace(-1.7,1.5,100)
plt.plot(x,U(x))
plt.plot(q_vec,U(q_vec),"x")
plt.show()
