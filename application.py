import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numba import njit,prange

# parameters
minibatch_size = 100
n_components = 100
obs_index = 65
K_exp1 = np.array([100,1000,10_000])

prior_sigma2 = 100

dt = 1e-2
nu = 1
nus_exp2 = np.array([1,10,100])
beta = 1


train_images = np.load("mnist/train-images.npy")
train_labels = np.load("mnist/train-labels.npy")
test_images = np.load("mnist/t10k-images.npy")
test_labels = np.load("mnist/t10k-labels.npy")

# keep only 7 and 9

train_images = train_images[(train_labels == 7) | (train_labels == 9)]
train_labels = (train_labels[(train_labels == 7) | (train_labels == 9)] == 9).astype(np.uint8)

N_obs = train_labels.shape[0]

test_images = test_images[(test_labels == 7) | (test_labels == 9)]
test_labels = (test_labels[(test_labels == 7) | (test_labels == 9)] == 9).astype(np.uint8)

pca = PCA(n_components = n_components, whiten = True)
pca.fit(train_images.reshape(N_obs,-1))

train_images_lowdim = pca.transform(train_images.reshape(N_obs,-1))
test_images_lowdim = pca.transform(test_images.reshape(test_images.shape[0],-1))


# # replicating fig 4 to test pca
# nb_number_plot = 10
# a = pca.inverse_transform(pca.transform(test_images.reshape(N_obs,-1)[:nb_number_plot])).reshape(nb_number_plot,28,28)
# 
# fig,axs = plt.subplots(2,nb_number_plot)
# for i in range(nb_number_plot):
#     axs[0,i].imshow(test_images[i],cmap='gray')
#     axs[1,i].imshow(a[i],cmap='gray')
# plt.show()
# 

@njit(inline="always")
def gradLLik(q):
    B = np.random.randint(0,N_obs,size = minibatch_size)
    x_B = train_images_lowdim[B]
    y_B = train_labels[B]
    x_Bq = x_B @ q
    # to avoid overflow : use log-sum-exp trick on positive values
    coefB = np.empty_like(x_Bq)
    coefB[x_Bq < 0] = y_B[x_Bq < 0] - np.exp(x_Bq[x_Bq < 0])/(1 + np.exp(x_Bq[x_Bq < 0]))
    coefB[x_Bq >= 0] = y_B[x_Bq >= 0] - 1 + np.exp(- x_Bq[x_Bq >= 0])/(1 + np.exp(- x_Bq[x_Bq >= 0]))
    somme = coefB @ x_B
    return -q/prior_sigma2 + (N_obs/minibatch_size) * somme

@njit(inline="always")
def odabado_step(p,q,zeta,nu):
    alpha1 = np.exp(- 0.5 * dt * zeta)
    p_half = alpha1 * p
    zeta_half = zeta + 0.5 * dt * (np.sum(p_half**2) - (p.shape[0]/beta))/nu
    q_half = q + 0.5 * dt * p_half
    gradUm = gradLLik(q_half)
    p_half_hat = p_half + dt * gradUm
    q_new = q_half + 0.5 * dt * p_half_hat
    zeta_new = zeta_half + 0.5 * dt * (np.sum(p_half_hat**2) - (p.shape[0]/beta))/nu
    alpha2 = np.exp(- 0.5 * dt * zeta_new)
    p_new = alpha2 * p
    return p_new,q_new,zeta_new

@njit(inline="always")
def integrate_q(K):
    zeta = 0
    p = np.random.normal(0,1,n_components)
    q = np.zeros(n_components)
    q_int = np.zeros(n_components)
    q2_int = np.zeros(n_components)
    for i in range(K):
        p,q,zeta = odabado_step(p,q,zeta,nu)
        q_int += q
        q2_int += q**2
    return (q_int / K, q2_int / K)

@njit(parallel=True)
def exp1(N):
    integrals_q = np.empty((3,N))
    integrals_q2 = np.empty((3,N))
    for i in prange(N):
        if (i%10 == 0):
            print(i)
        for j in range(3):
            q,q2 = integrate_q(K_exp1[j])
            integrals_q[j,i] = q[obs_index]
            integrals_q2[j,i] = q2[obs_index]
    return integrals_q,integrals_q2

@njit
def exp2():
    cum_mean = np.empty((3,100))
    avg_lik = np.empty((3,100))
    for i in range(3):
        nu = nus_exp2[i]
        zeta = 0
        p = np.random.normal(0,1,n_components)
        q = np.zeros(n_components)
        q_int = 0
        lik_int = 0
        for j in range(10_000):
            p,q,zeta = odabado_step(p,q,zeta,nu)
            q_int += q[obs_index]
            lik_int += np.mean(
                    np.exp(test_labels * (test_images_lowdim @ q)) /
                    (1 + np.exp(test_images_lowdim @ q)))
            if j%100 == 0:
                cum_mean[i,j//100] = q_int / (j+1)
                avg_lik[i,j//100] = lik_int / (j+1)
    return cum_mean,avg_lik

integrate_q(1)
exp1(1)
print("jit compilation done")

integrals_q,integrals_q2 = exp1(100)

fig,axs = plt.subplots(1,2,figsize=(10,5))
emp_mean_q = np.mean(integrals_q[2])
emp_var_q = np.var(integrals_q[2])
rescaled_q = ((np.sqrt(K_exp1*dt*1e-2/ emp_var_q)) * (integrals_q.T - emp_mean_q)).T
for i in range(3):
    axs[0].hist(rescaled_q[i],density=True,alpha=0.8)
x = np.linspace(-4,4)
axs[0].plot(x,(1/np.sqrt(2 * np.pi))*np.exp(-x**2/2))
axs[0].set_title("A")
axs[0].set_ylabel("EPDF")
axs[0].set_xlabel("Error")

emp_mean_q2 = np.mean(integrals_q2[2])
emp_var_q2 = np.var(integrals_q2[2])
rescaled_q2 = ((np.sqrt(K_exp1*dt*1e-2/ emp_var_q2)) * (integrals_q2.T - emp_mean_q2)).T
for i in range(3):
    axs[1].hist(rescaled_q2[i],density=True,alpha=0.8)
x = np.linspace(-4,4)
axs[1].plot(x,(1/np.sqrt(2 * np.pi))*np.exp(-x**2/2))
axs[1].set_title("B")
axs[1].set_xlabel("Error")
fig.savefig("fig/error_appl.pdf")
fig.savefig("fig/error_appl.png")
plt.show()


# cum_mean,avg_lik = exp2()
