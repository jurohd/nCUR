'''This is the module for sampling & Reconstruction'''

import numpy as np
import random
import math

from numpy import linalg as LA
from scipy.sparse.linalg import LinearOperator, bicgstab
from scipy.sparse import coo_matrix, csr_matrix

from ADMM_solver import splitted_ADMM

def col_noise(m, sig_c2):
	return np.random.normal(0,math.sqrt(sig_c2),(m,1))
	
def entry_noise(sig_e2):
	return np.random.normal(0,math.sqrt(sig_e2))

class Samples:
	def __init__( self, params, d ):
		self.targ_r			= params.targ_r
		self.d				= d
		self.f				= int(params.budget - d*params.pc)
		self.A				= params.A.data
		self.m, self.n 		= params.m, params.n
		self.sig_e2			= params.sig_e2
		self.sig_c2			= params.sig_c2
		self.omega_c 		= []
		self.omega_e 		= []
		self.X 				= np.zeros((self.m, self.n))
		self.Ctil   		= np.zeros((self.m, self.d))
		self.selected_col 	= []
		self.unselected_col = []
		self.A_obs			= np.zeros((self.m, self.n))
		self.selected_row 	= []
		self.obs_entries 	= None
		self.Z_reg			= None
		self.Ctil_reg		= None
		self.lbd			= params.lbd
		self.rho 			= params.rho
		
	def sample_col( self, repeat ):
#		repeat=1
		np.random.seed(int(self.d*self.lbd*self.rho*100*repeat)%2147483647)
		self.selected_col 	= np.sort( random.sample(range(0, self.n), self.d) )
		self.unselected_col = list( set( np.arange(self.n) ) - set( self.selected_col ) )
#		print(self.selected_col)
		f_count = self.f // self.n
#		print(self.d,f_count)
		self.selected_row = np.sort( np.random.permutation(self.m)[:f_count] )
		
		Ci = np.zeros((self.m, 1))
		for col in self.selected_col:
			c_col = self.A[:,col].reshape(self.m,1) + col_noise(self.m, self.sig_c2)
			Ci = np.hstack((Ci,c_col))
			for row in range(self.m):				#store sample coordinate for nnm
				if row not in self.selected_row:
					self.omega_c.append((row, col))
				
				
		self.Ctil = Ci[:,1:] 
		self.Ctil[self.selected_row,:] = 0
		self.X[:,self.selected_col] = self.Ctil
		self.A_obs = np.copy(self.X) 				#A_Observed for nnm
		self.Ctil_reg = self.Ctil[self.selected_row,:]
		self.Z_reg = np.zeros((min(f_count, self.m), self.n))
		
	def sample_entry( self ):
		
		ii = np.zeros(2*self.f)
		jj = np.zeros(2*self.f)
		ye = np.zeros(2*self.f)
		yeidx = 0 
		
		j = 0
		for col in range(self.n):
			i = 0
			for row in self.selected_row:
				entry = self.A[row, col] + entry_noise(self.sig_e2) #take samples
				self.Z_reg[i, j] = entry
				
				ii[yeidx] = row
				jj[yeidx] = col
				ye[yeidx] = entry
				self.A_obs[row, col] = entry
				
				yeidx += 1
				self.omega_e.append((row, col))
				i += 1
			j += 1
		
		ii = ii[:yeidx]
		jj = jj[:yeidx]
		ye = ye[:yeidx]
#		print(yeidx)
		self.obs_entries = coo_matrix((ye, (ii,jj)), shape=(self.m, self.n))
	
	def uniform_sampling( self ):
		ii = np.zeros(2*self.f)
		jj = np.zeros(2*self.f)
		ye = np.zeros(2*self.f)
		yeidx = 0 
		e_index_1d = np.sort(np.random.permutation(self.m * self.n)[:int(self.f)])
		
		for index in e_index_1d:
			m_2d = index // self.n 
			n_2d = index % self.n 
			entry = self.A[m_2d, n_2d] + entry_noise(self.sig_e2)
			
			ii[yeidx] = m_2d
			jj[yeidx] = n_2d
			ye[yeidx] = entry
			self.A_obs[m_2d, n_2d] = entry
			yeidx += 1
		
		ii = ii[:yeidx]
		jj = jj[:yeidx]
		ye = ye[:yeidx]
#		print(yeidx)
		self.obs_entries = coo_matrix((ye, (ii,jj)), shape=(self.m, self.n))
		
	def coh_sampling( self, est_samples ):
		U, s, Vt = np.linalg.svd(est_samples.A_obs,full_matrices=False)
		li_row = []
		v0 = 0
		for row in U[:,:self.targ_r]:
			li_row.append(LA.norm(row,2))
	
		for col in Vt[:self.targ_r,:]:
			li_v = LA.norm(col,2)
			if li_v > v0:
				v0 = LA.norm(col,2)
		
		Pij = np.zeros((self.m,self.n))
		for i in range(self.m):
			for j in range(self.n):
				Pij[i, j] = (li_row[i]+v0)*self.targ_r*(math.log10(self.n))/self.n
		
		Psum = np.sum(Pij)
		print(Psum)
		
		print("num entries:", self.f)
		c0 = self.f/Psum
		count = 0
		ii = np.zeros(2*self.f)
		jj = np.zeros(2*self.f)
		ye = np.zeros(2*self.f)
		yeidx = 0 
		
		ordered_pairs = []
		for i in range(self.m):
			for j in range(self.n):
				prob = random.random()
				if prob <= c0*Pij[i, j]:
					entry = self.A[i, j] + entry_noise(self.sig_e2)
					ii[yeidx] = i
					jj[yeidx] = j
					ye[yeidx] = entry
					self.A_obs[i, j] = entry
					yeidx += 1

		ii = ii[:yeidx]
		jj = jj[:yeidx]
		ye = ye[:yeidx]
#		print(yeidx)
		self.obs_entries = coo_matrix((ye, (ii,jj)), shape=(self.m, self.n))
						
		print("entries sampled #:", yeidx)
		
class Reconstruct:
	def __init__( self, params, sample, d ):
		self.hybrid_X 	= 0
		self.nnm_X		= 0
		self.nnmcvx_X	= 0
		self.d			= d
		
		if params.indic.show_hybrid:
			self.hybrid_X 	= Hybrid_recover( params, sample )
		if params.indic.show_nnm:
			
			self.nnm_X		= Nnm_recover( params, sample )
#			self.nnmcvx_X 	= Nnm_cvxpy( d, params, sample )
		
	
def Hybrid_recover(params, samples):
	I = np.eye(samples.d) 
#	print(params.lbd)
	C_basis = samples.Z_reg[:, samples.selected_col]
	XtX_lambda_I = C_basis.T@C_basis+params.lbd*I
	Xty = C_basis.T@samples.Z_reg
	w_lst = LA.lstsq(XtX_lambda_I,Xty,rcond=None)[0]
	samples.Ctil[samples.selected_row,:] = C_basis
	X = samples.Ctil@w_lst
	return X
	
def Nnm_recover(params, samples):
	rho = params.rho
	YC = samples.A_obs[:,samples.selected_col]
#	YC[samples.selected_row,:] = 0
	W1 = np.eye(params.m)
	W2 = np.eye(params.n)
	X, _, _ = splitted_ADMM(params.m, params.n, YC, math.sqrt(params.sig_c2), math.sqrt(params.sig_e2), samples.selected_col, samples.selected_row, samples.obs_entries, W1, W2, rho, maxIter = 1000)
	return X

def All_entry_nnm(param, length):
	nnm_allentrysum = 0
	nnm_list = []
	for i in range(param.repeat):
#		print(i)
		allentry_sample = Samples(param, 0)
		allentry_sample.uniform_sampling()
#		Nuc_e(allentry_sample.A_obs, indices, eps)
		X_allentry_nnm 	= Nnm_recover( param, allentry_sample )
		norm_allentry_X	= LA.norm(X_allentry_nnm-param.A.data,'fro')**2
		nnm_allentrysum += norm_allentry_X
		nnm_list.append(norm_allentry_X)
	nnmstd = np.std(nnm_list)
	avg = nnm_allentrysum/param.repeat
	norms_allentry 	= np.full(length, avg)
	norms_allmax 	= np.full(length, avg+2*nnmstd)
	norms_allmin 	= np.full(length, avg-2*nnmstd)
	return (norms_allentry, norms_allmax, norms_allmin)

def Twophase_nnm(param, length):
	nnm_2phasesum 	= 0
	nnm_list = []
	eta = 0.1 	#budget for levarage-score-estimating samples
	param.eta = eta
	for i in range(param.repeat):
#		print(i)
		estimating_param = param.copy_param()
		estimating_param.budget *= eta 
		estimating_samples = Samples(estimating_param, 0)
		estimating_samples.uniform_sampling()

		remaining_param = param.copy_param()
		remaining_param.budget *= (1 - eta) 
		remaining_samples = Samples(remaining_param, 0)

		remaining_samples.coh_sampling(estimating_samples)

		X_2phase_nnm 	= Nnm_recover( remaining_param, remaining_samples )
		norm_2pX		= LA.norm(X_2phase_nnm-param.A.data,'fro')**2
		nnm_2phasesum 	+= norm_2pX
		nnm_list.append(norm_2pX)
		
	nnmstd = np.std(nnm_list)
	avg = nnm_2phasesum/param.repeat
	
	norms_2phase 	= np.full(length, avg)
	norms_2pmax 	= np.full(length, avg+2*nnmstd)
	norms_2pmin    	= np.full(length, avg-2*nnmstd)
	return (norms_2phase, norms_2pmax, norms_2pmin)

#"""CVXPY sanity check"""
#def Nuc_norm_min( A_obs, omega_c, omega_e, eps_c, eps_e):
#	m,n = A_obs.shape
#	c_indices = tuple(zip(*omega_c))
#	e_indices = tuple(zip(*omega_e))
#	X = cp.Variable((m, n))
#	c_sum = cp.sum_squares(A_obs[c_indices]-X[c_indices])
#	e_sum = cp.sum_squares(A_obs[e_indices]-X[e_indices])
#	objective_fn = cp.norm(X, "nuc")
#	problem = cp.Problem(cp.Minimize(objective_fn), [c_sum <= eps_c, e_sum <= eps_e] )
#	problem.solve()
#	return X.value
#	
#def Nnm_cvxpy(d, params, samples):
#	eps_c = d * params.m * params.sig_c2
#	eps_e = len(samples.omega_e) * params.sig_e2
#	X = Nuc_norm_min(samples.A_obs, samples.omega_c, samples.omega_e, eps_c, eps_e)
#	return X
#	
#def Nuc_e( A_obs, indices, eps):
#	m,n = A_obs.shape
#	known_value_indices = tuple(zip(*indices))
#	X = cp.Variable((m, n))
#	fro_norm = cp.sum_squares(A_obs[known_value_indices]-X[known_value_indices])
#	objective_fn = cp.norm(X, "nuc")
#	problem = cp.Problem(cp.Minimize(objective_fn), [fro_norm <= eps] )
#	problem.solve()
#	return X.value
#
