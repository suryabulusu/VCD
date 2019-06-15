from densities import log2dGauss
import hmc

dim_z = 2
mu_v = np.random.normal(size = 2)
sigma_v = 0.1*np.ones(dim_z)

burnIters = 2
samplingIters = 1
adaptDuringBurn = 1
LF = 5

iters = 20000;  # Number of iterations

# RMSProp parameters
rhomu = 0.01;
rhosigma = 0.005;
ReducedBy = 0.9; 
ReducedEvery = 2000;
TT = 1;
kappa0 = 0.1;

# Control variates
control_variate = 0;
theta = 0.9;
use_control_variate = 1;

# Quantities to compute at each iteration
stochasticDiv = np.zeros(iters);
ELBO_q = np.zeros(iters);
expLogLik_qt = np.zeros(iters);
first_term = np.zeros(iters);

acceptHist = np.zeros(burnIters+samplingIters);
acceptRate = 0;

# vars for RMSProp
Gt_mu = 0.01*np.ones(np.shape(mu_v)[0]);
Gt_sigma = 0.01*np.ones(np.shape(sigma_v));

# Algorithm

dist = log2dGauss(mu, L)

for i in range(iters):
  eta = np.random.normal(size = 2)
  z = mu_v + sigma_v @ eta
  log_pxz, grads = dist.eval(z)
  
  precond_mu_first = grads
  precond_sigma_first = grads @ eta;
  first_term[i] = log_pxz + 0.5*dim_z
  
  zt, samples, extras = hmc(z, dist, 0.5/dim_z, burnIters, samplingIters, adaptDuringBurn, LF)
  accRate, accHist, log_densHist, delta = extras
  # Keep track of acceptance rate (for information purposes only)
  acceptHist += accHist/iters
  acceptRate += accRate/iters
  # Adapt the stepsize for HMC
  eps = delta

  # Evaluate the densities and the gradient (second expectation)
  logp_xzt, gradzt = dist.eval(zt);
  diff = (zt - mu_v)/sigma_v;
  diff2 = diff**2;
  f_zt = logp_xzt + 0.5*np.sum(diff2);
  precond_mu_second = (eta/sigma_v) * (f_zt - control_variate);
  precond_sigma_second = (-1/sigma_v + (eta**2) / sigma_v) * (f_zt - control_variate);

  # Evaluate the gradient (third expectation)
  precond_mu_third = diff / sigma_v;
  precond_sigma_third = diff2 / sigma_v;

  # Total gradient
  grad_mu = precond_mu_first - precond_mu_second + precond_mu_third;
  grad_sigma = precond_sigma_first - precond_sigma_second + precond_sigma_third; 
  
  # RMSprop updates
  kappa = kappa0
  if (i == 0): kappa = 1
  Gt_mu = kappa*(grad_mu**2) + (1-kappa)*Gt_mu;
  mu_v += rhomu*grad_mu / (TT + np.sqrt(Gt_mu));
  Gt_sigma = kappa*(grad_sigma**2) + (1-kappa)*Gt_sigma;
  sigma_v += rhosigma*grad_sigma / (TT + np.sqrt(Gt_sigma));
  sigma_v[sigma_v < 0.00001] = 0.00001; # for numerical stability
  
  entropy = 0.5*dim_z*np.log(2*np.pi) + np.sum(np.log(sigma_v)) + dim_z/2;
  ELBO_q[i] = np.mean(log_pxz) + np.mean(entropy);
  if (i+1) % 1000 == 0:
      print("Iter=%d, ELBO_q=%f", i, ELBO_q[i]);

  if (i+1) % ReducedEvery == 0:
      rhosigma = rhosigma*ReducedBy;
      rhomu = rhomu*ReducedBy;
