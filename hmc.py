def hmc(current_q, log_pxzdens, eps, Burn, T, adapt, L):
  n = np.shape(current_q)[0]
  acceptHist = np.zeros(Burn + T)
  log_densHist = np.zeros(Burn + T)
  
  samples = np.zeros([T, n])
  
  if Burn + T == 0:
    z = current_q
    return z, samples, 0, acceptHist, log_densHist, eps
  
  # useful for adapting eps
  eta = 0.01
  opt = 0.7
  
  cnt = 0
  
  for i in range(Burn + T):
    q = current_q
    p = np.random.normal(size = 2)
    
    current_p = p
    logpxz, grads = log_pxzdens.eval(q)
    
    #print(current_q, p)
    
    # half step
    curr_U = - logpxz
    grad_U = - grads
    p = p - eps*grad_U/2;
    
    for j in range(L):
      q = q + eps*p
      if j != L - 1:
        logpxz, grads = log_pxzdens.eval(q)
        prop_U = -logpxz
        grad_U = -grads
        p = p - eps*grad_U
        
    # half step momentum
    logpxz, grads = log_pxzdens.eval(q)
    prop_U = -logpxz
    grad_U = -grads
    p = p - eps*grad_U
    p = -p # negate mom at the end of trajectory to make proposal symmetric
    
    # evaluate Pot and Kin energies
    curr_K = np.sum(current_p**2)/2
    prop_K = np.sum(p**2)/2
    #print(curr_U, prop_U, curr_K, prop_K)
    if (np.random.rand() < np.exp(curr_U - prop_U + curr_K - prop_K)):
      accept = 1
      #print("yo mach")
    else:
      accept = 0
      
    acceptHist[i] = accept
    
    if accept == 1:
      current_q = q
      curr_U = prop_U
      
    if i <= Burn and adapt == 1:
      eps = eps + eta*((accept-opt)/opt)*eps
    else:
      cnt += 1
      samples[cnt,:] = current_q
      
    log_densHist[i] = -curr_U
      
  z = current_q
  accRate = np.mean(acceptHist)
  
  return z, samples, (accRate, acceptHist, log_densHist, eps)
