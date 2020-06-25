import logging
import math
import numpy as np
import scipy.misc
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab
from scipy.linalg import svd, norm, pinv, diagsvd
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
#handler = logging.StreamHandler()
#handler.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')
#handler.setFormatter(formatter)
#logger.handlers = [handler]

def Shrink(M, tau):
	"""compute prox of tau*||M||_nuclear"""
#	print("enter shrink")
#	M=np.where(np.isfinite(M), M, 1e-10)
	U, s, Vh = svd(M, full_matrices=False,check_finite=False)
	s = (s - tau)
	s[s < 0] = 0
	return U*s @ Vh

def Lyapunov_solve(A, B, C, Y):
	"""Solves A.*.X + B*X*C = Y, where .*. indicates Hadamard product"""
	
	#logging.debug("Entering lyapunov solver")
	Binv = pinv(B)
	Cinv = pinv(C)
	(m,n) = A.shape
	unvec = lambda x: np.reshape(x, (m,n))
	vec = lambda X : np.reshape(X, (m*n,))
	Sys = lambda x : np.multiply(A, unvec(x)) + B @ unvec(x) @ C
	precond = lambda x : Binv @ unvec(x) @ Cinv
	L = LinearOperator((m*n, m*n), matvec=Sys, rmatvec=Sys)
	precondop = LinearOperator((m*n, m*n), matvec=precond)
	
	x, _ = bicgstab(L, vec(Y), M=precondop, tol=1e-14, atol=1e-14)
	X = unvec(x)
	#print( np.linalg.norm(A*X + B.dot(X.dot(C)) - Y)/np.linalg.norm(Y))
	#logging.debug("Exiting Lyapunov solver")
	return X

"""ADMM solver for 
			  argmin_X ||Z||_nuclear + lambda/2*||A(X) - y||_2^2 subject to Z = W1*X*W2
	   where yscaled = [ycol/sigma_col; ye/sigma_e] 
	   and A(X) = [Acol(X)/sigma_col; Ae(X)/sigma_e]
	"""

def reweighted_ADMM(m: "rows", 
					n: "columns", 
					YC: "observed columns", 
					sc : "sigma_col", 
					se : "sigma_e", 
					observed_columns : "vector of column indices that were observed", 
					observed_entries : "sparse entry observation mask (entries=observations)", 
					W1: "left reweighing matrix for nuclear norm", 
					W2: "right reweighing matrix for nuclear norm", 
					lambdaa: "measurement error penalty parameter", 
					rho: "penalty parameter for ADMM" = 2, 
					maxIter : "maximum number of iterations of ADMM" = 1000,
					ptol : "absolute tolerance for primal residual" = .001,
					dtol : "absoulte tolerance for dual residual" = .001):
	
	i, j, ye = scipy.sparse.find(observed_entries)
	Omegae = coo_matrix(([1]*len(i), (i, j)), shape=(m,n))
	WW1 = W1.T @ W1
	WW2 = W2 @ W2.T 
	
	Mcol = np.zeros((m,n))
	Mcol[:,observed_columns] = 1
	M = Omegae.toarray()/se + Mcol/sc
	scaledM = Omegae.toarray()/se**2 + Mcol/sc**2

	Adjyscaled = np.zeros((m,n))
	Adjy = np.zeros((m,n))
	for colidx in range(len(observed_columns)):
		Adjyscaled[:, observed_columns[colidx]] += YC[:, colidx]/pow(sc,2)
		Adjy[:, observed_columns[colidx]] += YC[:, colidx]
	for entryidx in range(len(i)):
		Adjyscaled[i[entryidx], j[entryidx]] += ye[entryidx]/pow(se,2)
		Adjy[i[entryidx], j[entryidx]] += ye[entryidx]

	Xold = Adjyscaled
	Zold = W1 @ Xold @ W2
	Thetaold = np.zeros_like(Zold)
	primal_resid = 0
	dual_resid = np.Inf

	mu = 10
	tau_increase = 2
	tau_decrease = 2
	

	for t in range(maxIter):
		R = lambdaa * Adjyscaled + rho*W1.T @ (Zold + 1/rho*Thetaold) @ W2.T
		Xnew = R/(lambdaa*scaledM+rho)
#		Xnew = Lyapunov_solve(lambdaa*scaledM, rho*WW1, WW2, R)
		Znew = Shrink(W1 @ Xnew @ W2 - 1/rho*Thetaold, 1/rho)
		Thetanew = Thetaold + rho*(Znew - W1 @ Xnew @ W2)
		
		primal_resid = norm( Znew - W1 @ Xnew @ W2, 'fro')
		dual_resid = norm( rho * W1.T @ (Zold - Znew) @ W2.T, 'fro')
		nll = norm(np.multiply(M, Xnew - Adjy), 'fro')**2
#		logging.info(f"({t+1}/{maxIter}) primal residue: {primal_resid}, dual residue: {dual_resid}, NLL: {nll}")
		if primal_resid > mu*dual_resid:
			rho *= tau_increase
		elif dual_resid > mu*primal_resid:
			rho /= tau_decrease
		
		Xold = Xnew
		Zold = Znew
		Thetaold = Thetanew
		
		if primal_resid < ptol and dual_resid < dtol:
			return Xold, Zold, Thetaold
	return Xold, Zold, Thetaold
def splitted_ADMM(m: "rows", 
				n: "columns", 
				YC: "observed columns", 
				sc : "sigma_col", 
				se : "sigma_e", 
				observed_columns : "vector of column indices that were observed", 
				selected_rows:	"rows that is replaced by entries",
				observed_entries : "sparse entry observation mask (entries=observations)", 
				W1: "left reweighing matrix for nuclear norm", 
				W2: "right reweighing matrix for nuclear norm", 
				lambdaa: "measurement error penalty parameter", 
				rho: "penalty parameter for ADMM" = 2, 
				maxIter : "maximum number of iterations of ADMM" = 100,
				ptol : "absolute tolerance for primal residual" = .0001,
				dtol : "absolute tolerance for dual residual" = .0001):
	"""if lambdaa is a tuple (lambda1, lambda2), then ADMM solver for
				 argmin_X ||Z||_nuclear + lambda1/2*||A1(X) - ycolscaled||_2^2 
																+ lambda2/2*||A1(X) - yescaled||_2^2 
				 subject to Z = W1*X*W2
			where ycolscaled = [ycol/sigma_col], yescaled = [ye/sigma_e],
			and A1(X) = [Acol(X)/sigma_col], A2(X) = [Ae(X)/sigma_e]
			
			if lambdaa is a scalar, then lambda1 = lambda2 = lambdaa
	"""
	i, j, ye = scipy.sparse.find(observed_entries)
	Entry_observation_mask = coo_matrix(([1]*len(i), (i, j)), shape=(m,n))
	
	if type(lambdaa) is tuple or type(lambdaa) is list:
		lambda1 = lambdaa[0]
		lambda2 = lambdaa[1]
	else:
		lambda1 = lambdaa
		lambda2 = lambdaa
#	WW1 = W1.T @ W1
#	WW2 = W2 @ W2.T
	
	Column_observation_mask = np.zeros((m,n))
	Column_observation_mask[:,observed_columns] = 1
	Column_observation_mask[selected_rows,:] = 0
	Observation_mask = Entry_observation_mask.toarray()/se + Column_observation_mask/sc
	Scaled_observation_mask = lambda1 * Column_observation_mask/sc**2 + \
														lambda2 * Entry_observation_mask.toarray()/se**2                       
	
	Observations = np.zeros((m,n))
	Scaled_column_observations = np.zeros((m,n))
	Scaled_entry_observations = np.zeros((m,n))
	for colidx in range(len(observed_columns)):
		Scaled_column_observations[:, observed_columns[colidx]] += YC[:, colidx]/pow(sc,2)
		Observations[:, observed_columns[colidx]] += YC[:, colidx]
	Observations[selected_rows,:] = 0
	Scaled_column_observations[selected_rows,:] = 0
	
	for entryidx in range(len(i)):
		Scaled_entry_observations[i[entryidx], j[entryidx]] += ye[entryidx]/pow(se,2)
		Observations[i[entryidx], j[entryidx]] += ye[entryidx]
	
	Xold = Observations
#	Zold = W1 @ Xold @ W2
	Zold = Xold
	Thetaold = np.zeros_like(Zold)
	primal_resid = 0
	dual_resid = np.Inf

	mu = 10
	tau_increase = 2
	tau_decrease = 2
		
	for t in range(maxIter):
#		print("itr=",t)
#		R = lambda1 * Scaled_column_observations + lambda2 * Scaled_entry_observations + \
#				rho*W1.T @ (Zold + 1/rho*Thetaold) @ W2.T
		R = lambda1 * Scaled_column_observations + lambda2 * Scaled_entry_observations + \
				rho* (Zold + 1/rho*Thetaold)
		Xnew = R/(Scaled_observation_mask+rho)
#		Xnew = Lyapunov_solve(Scaled_observation_mask, rho*WW1, WW2, R)
#		Znew = Shrink(W1 @ Xnew @ W2 - 1/rho*Thetaold, 1/rho)
#		Thetanew = Thetaold + rho*(Znew - W1 @ Xnew @ W2)
		diff_M = Xnew - 1/rho*Thetaold
#		print('rho =',rho)
		try:
			Znew = Shrink(diff_M, 1/rho)
		except LinAlgError:
			print("gethere")
			return Xold, Zold, Thetaold
		else:
			Thetanew = Thetaold + rho*(Znew - Xnew)
		
#		primal_resid = norm( Znew - W1 @ Xnew @ W2, 'fro')
		primal_resid = norm( Znew - Xnew, 'fro')
#		dual_resid = norm( rho * W1.T @ (Zold - Znew) @ W2.T, 'fro')
		dual_resid = norm( rho * (Zold - Znew) , 'fro')
		nll = norm(np.multiply(Observation_mask, Xnew - Observations), 'fro')**2
#		logging.info(f"({t+1}/{maxIter}) primal residue: {primal_resid}, dual residue: {dual_resid}, NLL: {nll}")

		if primal_resid > mu*dual_resid:
			rho *= tau_increase
		elif dual_resid > mu*primal_resid:
			rho /= tau_decrease
		
		Xold = Xnew
		Zold = Znew
		Thetaold = Thetanew
	
		if primal_resid < ptol and dual_resid < dtol:
			return Xold, Zold, Thetaold
			
	return Xold, Zold, Thetaold
## Test ADMM solver for reweighed NN minimization
#m, n = im.shape
#d = 2*r
#f = 2*(m*r + n*r) - n*d
#
## use col noise with PSNR of 50 decibels (can't compute SNRs because don't know matrix A, so can't use its Frob norm)
#
## use entry noise with PSNR of 70 decibels
#sc = math.pow(10, -5/2)
#se = math.pow(10, -7/2)
#
#observed_columns = np.random.choice(n, size=d, replace=False)
#YC = A[:, observed_columns] + sc*np.random.randn(m, d)
#
#sampleprob = f/(m*(n-d))
#
#i = [-1]*(2*f)
#j = [-1]*(2*f)
#ye = np.array([-1.0]*(2*f))
#
#curoffset = 0
#for rowidx in range(m):
#	for colidx in range(n):
#		bern = np.random.rand() 
#		if (bern < sampleprob) and (colidx not in observed_columns):
#			i[curoffset] = rowidx
#			j[curoffset] = colidx
#			ye[curoffset] = A[rowidx, colidx]
#			curoffset += 1
#i = i[:curoffset]
#j = j[:curoffset]
#ye = ye[:curoffset] + se*np.random.randn(len(i),)
#print(curoffset, f)
#observed_entries = coo_matrix((ye, (i,j)), shape=(m,n))
#
#W1 = np.eye(m)
#W2 = np.eye(n)
#lambdaa = 10
#
#Arecovered, Z, Theta = ADMM_solve(m, n, YC, sc, se, observed_columns, observed_entries, W1, W2, lambdaa, maxIter=100)