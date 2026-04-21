#import "@preview/touying:0.7.1": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
)

= Hypocoercivity of Adaptive Langevin Dynamics

#cite(<leimkuhler2023hypocoercivitypropertiesadaptivelangevin>,form:"author")

#v(40pt)

Molecular Simulation homework by Jules Herrmann

== Langevin Dynamics

$
dif q & = M^(-1) p dif t \

dif p & = (- nabla U(q) - zeta M^(-1) p) dif t + sigma dif W
$

- $M$ : mass matrix
- $zeta$ : friction
- $sigma$ : strengh of thermal noise

Invariant distribution = Boltzmann-Gibbs distribution $p_beta prop e^(-beta U)$

With $beta = (2 zeta)/sigma^2$

== Thermostat

$nabla U$ can be hard to compute.

We may only have access to a *noisy* approximation $hat(nabla) U$

#v(40pt)

Temperature of the system $T = 1/(k_B "Tr"(M^(-1))) EE(p^T M^(-2) p)$ must be constant.

Noise in $hat(nabla) U$ $->$ drift of system temperature $->$ incorrect sampling

Solution : Control temperature with a thermostat @ding2014bayesian

== Thermostat

Friction $zeta$ reinterpreted as a dynamical variable

$
dif zeta = 1/nu (p^T M^(-2) p - 1/beta "Tr"(M^(-1))) $


#grid(columns:(50%,50%),[
Temperature too low :
- $p^T M^(-2) p < 1/beta "Tr"(M^(-1))$
- reduced friction
- less energy is dissipated
- temperature increases
],[
Temperature too high :
- $p^T M^(-2) p > 1/beta "Tr"(M^(-1))$
- increased friction
- more energy is dissipated
- temperature decreases
])

== Adaptive Langevin Dynamics

$
dif q & = M^(-1) p dif t \

dif p & = (- nabla U(q) - zeta M^(-1) p) dif t + sigma_G dif W_G + sigma_A dif W_A \

dif zeta & = 1/nu (p^T M^(-2) p - 1/beta "Tr"(M^(-1)))
$

- $sigma_G$ is the unknown noise of gradient approximation
- $sigma_A$ is the known thermal noise
- $beta$ is the prescribed inverse temperature
- $nu$ is the strengh of the thermostat

== Invariant distribution

$
pi(dif q,dif p,dif zeta) = Z^(-1) exp(1/beta [(p^T M^(-1) p)/2 + U(q) + nu/2 (zeta - gamma)^2]) dif q dif p dif zeta
$

Where
- $Z$ is such that this distribution is a probability measure
- $gamma = (beta (sigma_A^2 + sigma_G^2))/2$

== Normalized Dynamics

Change of variable
$epsilon & = sqrt(nu)$ and $xi & = sqrt(nu) (zeta - gamma)$

$
dif q & = p dif t \

dif p & = (- nabla U(q) - (xi/epsilon + gamma) p) dif t + sqrt((2 gamma)/beta) dif W \

dif xi & = 1/epsilon (p^T p - n/beta)
$

== Normalized Dynamics

New normalized invariant distribution

$
pi(dif q,dif p,dif xi) = Z^(-1) exp(1/beta [(p^T p)/2 + U(q) + xi^2 /2]) dif q dif p dif xi
$

Independant from $gamma$ and $epsilon$ : we can more easily study the effect of the parameters of the system on its convergence

== Trajectory average

Langevin dynamics are used to estimate $EE_pi (phi)$ with the trajectory average estimator $hat(phi)_t$

$
hat(phi)_t = 1/t integral_0^t phi(q_s,p_s,xi_s) dif s
$

$pi$ is invariant, the system is irreducible and regular :

$
hat(phi)_t -->_(t -> inf)^"a.s." EE_pi (phi)
$

== Central Limit Theorem

$
sqrt(t) (hat(phi)_t - EE_pi (phi) ) -->_(t->infinity)^"law" cal(N)(0, sigma^2_(epsilon,gamma)(phi))
$

And there exist $C$ and $lambda$, s.t.
$
sigma^2_(epsilon,gamma)(phi) <= (2 C ||phi||^2_(L^2(pi)))/lambda "max"(gamma,epsilon^2 / gamma, gamma epsilon^2 , 1/ (gamma epsilon^2))
$

== Numerical Integrator


#align(center,table(
  inset: 10pt,
  columns: 2,
  table.header(
    [Generator],
    [Single Numerical Step]
  ),
  $cal(L)_A = p dot nabla_q$,
  $ Phi^A_(Delta_t)(q)= q + Delta t p$,
  $cal(L)_B = - nabla U(q) dot nabla_p$,
  $ Phi^B_(Delta_t)(p) = p - Delta t nabla U(q)$,
  $cal(L)_D = 1/epsilon ( |p|^2 - n/beta) partial_xi$,
  $ Phi^D_(Delta_t)(xi) = xi + (Delta t)/epsilon ( |p|^2 - n/beta)$,
  $cal(L)_O = -zeta p dot nabla_p + sigma^2 laplace_p$,
  $ Phi^O_(Delta_t)(p) = e^(- Delta t zeta) p +  G(sigma,zeta,Delta t) R$
))

Where $R ~ cal(N)(0,1)$ and
$G(sigma,zeta,Delta t) =
cases(sigma sqrt((1 - e^(-2 Delta t zeta))/(2 zeta)) & "if" zeta != 0,
sigma sqrt(Delta t) & "if" zeta = 0)$

== Numerical Integrator

- BADODAB
$
Phi^B_((Delta t)/2) compose
Phi^A_((Delta t)/2) compose
Phi^D_((Delta t)/2) compose
Phi^O_(Delta t) compose
Phi^D_((Delta t)/2) compose
Phi^A_((Delta t)/2) compose
Phi^B_((Delta t)/2)
$

- ODABADO
$
Phi^O_((Delta t)/2) compose
Phi^D_((Delta t)/2) compose
Phi^A_((Delta t)/2) compose
Phi^B_(Delta t) compose
Phi^A_((Delta t)/2) compose
Phi^D_((Delta t)/2) compose
Phi^O_((Delta t)/2)
$




== CLT on toy example

$U(q) = (q^2 - 1)^2 + 1/2 q$

$
hat(phi)_K = 1/K sum_(k=0)^(K-1) phi(q^((k)),p^((k)),xi^((k)))
$

$
dash(phi)_(K,N) = 1/N sum_(n=1)^(N) hat(phi)_K^((n))
$

$
hat(sigma)^2 = 1/N sum_(n=1)^N (hat(phi)_K^((n)) - dash(phi)_(K,N))
$


==

#figure(
  image("fig/fig_variance.pdf",width:80%),
  caption : [Variance of different observable (Right : $gamma = 1$, Left : $epsilon = 1$)])

==

#figure(
  image("fig/epdf.pdf",width:70%),
  caption : [Empirical distribution of rescaled residual error at different times])

== Application to Bayesian Logistic Regression with a Stochastic gradient

Observation : $y_j in {0,1}$, $x_j in RR^d$ for $j in [0,tilde(N)]$

Parameter : $q in RR^d$

Prior : $q ~ cal(N)(0,sigma^2)$ i.e. $p_0(q) prop exp(- (|q|^2)/sigma)$

Likelihood : $p((y_j),(x_j) | q) = limits(product)_j (exp(y_j <x_j, q>))/(1 + exp(<x_j, q>))$

Posterior : $p(q | (y_j) , (x_j)) prop p_0(q) space p((y_j),(x_j) | q)$

==

If we choose $-nabla U(q) = nabla log(p_0(q)) + nabla log(p((y_j),(x_j) | q))$

Then, the invariant distribution is the posterior distribution

$
pi(dif q) prop exp(-U(q)) dif q prop p(q | y_j , x_j) dif q
$

$- nabla U(q)$ expensive to compute : use of a stochastic gradient

$
- tilde(nabla)U = nabla log(p_0(q)) + tilde(N)/m sum_(j in B) nabla log(p(y_j,x_j | q))
$

With $B$ a random batch of size $m$ of observations

==

Application to MNIST dataset, projected on a smaller space using a Principal Component Analysis

#figure(
  image("fig/pca_example.pdf",width:100%),
  caption : [Examples of images from the MNIST dataset and their projection on the 100 principal components])


==

#figure(
  image("fig/error_appl.pdf",width:80%),
  caption : [Empirical distribution of error (Right : $q_65$, Left : $q_65^2$)])

==

#figure(
  image("fig/time_evolution.pdf",width:80%),
  caption : [Evolution of $EE(q_65)$ and of the average likelihood over time])

#bibliography("refs.bib",title : none)
