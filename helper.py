'''This is the module for dummy functions and variables'''
import numpy as np
import copy
import math
from algorithm import Samples, Reconstruct
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D

class Indicator:
	def __init__(self, data_type, show_hybrid, show_nnm, show_nnm_e, show_2phase, h_budget, h_noise ):
		
		self.show_nnm 		= show_nnm
		self.show_nnm_e		= show_nnm_e
		self.show_2phase 	= show_2phase
		self.show_hybrid 	= show_hybrid
		self.data_type 		= data_type
		self.h_budget 		= h_budget
		self.h_noise 		= h_noise
		
class Matrix: 
	
	def __init__(self, A): 		# m by n matrix
		self.data			= A
		self.m 				= A.shape[0]
		self.n 				= A.shape[1]
		self.rank			= LA.matrix_rank(A)
		self.stable_rank	= None				# default if not calculated
		self.entry_mean		= np.mean(A)
		self.entry_var		= np.var(A)
		
		self.U				= None
		self.s				= np.full(self.rank, 0)
		self.Vt				= None
		
	def get_reducedSVD(self):
		self.U, self.s, self.Vt = LA.svd(self.data, full_matrices = False)
		return self.U, self.s, self.Vt
	
	def calc_stable_r(self):
		if self.s[0] == 0:
			self.get_reducedSVD()
		self.stable_rank = np.around(np.sum(self.s**2)/(self.s[0]**2),decimals = 4)
		return self.stable_rank
		
	def rank_r_proj(self, r):
		if self.s[0] == 0:
			self.get_reducedSVD()
			
		self.rank 			= r
		self.U				= self.U[:,:r]
		self.s				= self.s[:r]
		self.Vt				= self.Vt[:r,:]
		self.data 			= self.U*self.s@self.Vt
		
		self.entry_mean		= np.mean(self.data)
		self.entry_var		= np.var(self.data)
	
	def copy_mat(self):
		return copy.copy(self)
		
class Params:
	'''
	targ_r: 		target rank that we want to approximate
	pe: 			cost for sampling a entry
	pc:				cost for sampling a column
	gamma_coeff: 	coeff for determine the entry noise and column noise
	gamma:			ratio between 
	sig_e2:			entrywise noise variance
	sig_c2:			columnwise noise variance
	budget:			budget for sampling
	lbd:			regularization parameter
	d_interval:		output plot d interval
	repeat:			experiment repeat times
	'''
	def __init__( self, indic, is_synthetic, matrix_info, h_budget, h_noise, targ_r, alpha ):
		self.indic			= indic
		self.m, self.n 		= matrix_info.m, matrix_info.n
		self.A				= matrix_info
		self.targ_r			= targ_r
		self.pe 			= 1
		self.alpha			= alpha
		self.beta 			= 0
		self.pc 			= matrix_info.m * self.pe * alpha
		self.standard_err	= -1
		self.maxnum_cols	= 0
		self.rho 			= 10
		self.repeat  		= 1
		
		if is_synthetic == "synthetic":
			self.c0 		= 3
			self.d_interval = 1
			self.gamma_coef	= 1
			self.repeat		= 10
			self.lplotcoeff = 4
			self.rplotcoeff = 4
			if h_noise:
				self.sig_e2 = 0.2**2
				self.lbd	= 2.15
			else:
				self.sig_e2 = 0.1**2
				self.lbd	= 0.00145
		
		if is_synthetic == "movielens":
			self.c0 		= 10
			self.d_interval = 20
			self.gamma_coef	= 5
			self.repeat		= 10
			self.lplotcoeff = 2
			self.rplotcoeff = 2
			
			if h_noise:
				self.sig_e2 = 0.15**2
				self.lbd	= 81
			else:
				self.sig_e2 = 0.05**2
				self.lbd	= 23
		
		if is_synthetic == "jester":
			self.c0 		= 1.1
			self.d_interval = 1
			self.gamma_coef	= 10
			self.repeat		= 10
			self.lplotcoeff = 2.2
			self.rplotcoeff = 0.3
			
			if h_noise:
				self.sig_e2 = 0.5**2
				self.lbd	= 11124
			else:
				self.sig_e2 = 0.2**2
				self.lbd	= 80
			
		if h_budget:
			self.budget 	= self.c0 * matrix_info.m * self.targ_r * self.pe
		else:
			self.budget		= self.c0 * self.targ_r * self.pc + matrix_info.n * self.targ_r * self.pe
									
		self.gamma = math.ceil(self.gamma_coef/self.alpha)
		self.sig_c2 = self.sig_e2*self.gamma
		
		self.print_param() #print param
		
	def copy_param(self):
		return copy.copy(self)

	def print_param(self):
		maxnum_entries 		= math.floor(self.budget//self.pe)
		self.beta 			= maxnum_entries/(self.m*self.n)
		self.maxnum_cols 	= min(math.floor(self.budget//self.pc),self.n)
#		print(int(self.lplotcoeff*self.targ_r), int(self.maxnum_cols-self.rplotcoeff*self.targ_r))
		xmax = int(self.maxnum_cols-self.rplotcoeff*self.targ_r)
		xmin = int(self.lplotcoeff*self.targ_r)
		self.xticksInt		= np.linspace( xmin, xmax,(xmax-xmin)//int(self.d_interval*2), dtype = 'int16')
		A_r 				= self.A.copy_mat()
		A_r.rank_r_proj(self.targ_r)
		self.standard_err = LA.norm(self.A.data-A_r.data, 'fro')**2
		
		print('size of A is', self.m,'x', self.n)
		print('maximum column could be sampled =',self.maxnum_cols)
		print('beta =',self.beta)
		print('|A-Ar|_F^2 =',self.standard_err)
#		print('average err per entry =',math.sqrt(self.standard_err)/self.m/self.n)
		print("entriwise SNR_db=",np.mean((self.A.data)**2)/(self.sig_c2)) 
		
	def print_budget(self):
		self.maxnum_cols 	= min(math.floor(self.budget//self.pc),self.n)
		maxnum_entries 		= math.floor(self.budget//self.pe)
		self.beta 			= maxnum_entries/(self.m*self.n)
		print('beta =',self.beta)
		
class Plot_params:
	def __init__( self, d_axis):
		self.d_axis = d_axis
		self.length = len(d_axis)
		self.all_e 	= np.zeros(self.length)
		self.allmax = np.zeros(self.length)
		self.allmin = np.zeros(self.length)
		self.phase2 = np.zeros(self.length)
		self.p2max 	= np.zeros(self.length)
		self.p2min 	= np.zeros(self.length)
		self.hyb	= np.zeros(self.length)
		self.hybmax	= np.zeros(self.length)
		self.hybmin	= np.zeros(self.length)
		self.nnm	= np.zeros(self.length)
		self.nnmmax = np.zeros(self.length)
		self.nnmmin = np.zeros(self.length)
		
	def set_all_entry( self, triplet ):
		self.all_e 	= triplet[0]
		self.allmax = triplet[1]
		self.allmin = triplet[2]
		
	def set_phase2( self, triplet ):
		self.phase2 = triplet[0]
		self.p2max 	= triplet[1]
		self.p2min 	= triplet[2]
		
	def set_hyb_nnm( self, sextuplets ):
		self.hyb	= sextuplets[0]	
		self.hybmax	= sextuplets[1]
		self.hybmin	= sextuplets[2]
		self.nnm	= sextuplets[3]
		self.nnmmax = sextuplets[4]
		self.nnmmin = sextuplets[5]
	
	def plotgraph(self, param, method_indicator):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)
		ax.set_xlabel('d')
		ax.set_ylabel(r'$\frac{ \Vert X-A \Vert^2_F}{ \Vert A_r-A \Vert^2_F}$')

		legendlabel = []
		legendline = []
		if param.indic.show_hybrid:
			hybmid, = ax.plot(self.d_axis, self.hyb/param.standard_err, c='b',label='Hybrid method' )
			hybshaded = ax.fill_between(self.d_axis,self.hybmin/param.standard_err,self.hybmax/param.standard_err, alpha=0.2  )
			legendline.append((hybmid,hybshaded))
			legendlabel.append("HReg")


		if param.indic.show_nnm:	
			nnmmid, = ax.plot(self.d_axis, self.nnm/param.standard_err, c='r' )
			nnmshaded = ax.fill_between(self.d_axis,self.nnmmin/param.standard_err,self.nnmmax/param.standard_err, color = 'orange', alpha=0.2  )
			legendline.append((nnmmid, nnmshaded))
			legendlabel.append("NNs")
		#		ax.plot(d_axis, norms_cvxnnm/param.standard_err, c='g',label='Nuclear Norm cvxpy' )
		if param.indic.show_nnm_e:
			nnme, = ax.plot(self.d_axis, self.all_e/param.standard_err, c='purple' )
			nnmeshaded = ax.fill_between(self.d_axis,self.allmin/param.standard_err,self.allmax/param.standard_err, color = 'orchid', alpha=0.1  )
			legendline.append((nnme, nnmeshaded))
			legendlabel.append('NNa')
		if param.indic.show_2phase:
			nnm2p, = ax.plot(self.d_axis, self.phase2/param.standard_err, c='yellow' )
			nnm2pshaded = ax.fill_between(self.d_axis,self.p2min/param.standard_err,self.p2max/param.standard_err, color = 'lightyellow', alpha=0.5  )
			legendline.append((nnm2p,nnm2pshaded))
			legendlabel.append('Chen')
		print(legendlabel)
		ax.legend(legendline, legendlabel, loc='upper right')

		#	plt.show()
		noisestring = "highnoise"
		if method_indicator.h_noise == False:
			noisestring = "lownoise"
			
		budgetstr = "highbudget"
		if method_indicator.h_budget == False:
			noisestring = "lowbudget"
		plt.savefig('./CV/'+method_indicator.data_type+'/'+noisestring+'/rho='+str(param.rho)+'.png',dpi=300)
		np.savez('./numpy/'+method_indicator.data_type+'_'+budgetstr+'_'+noisestring+'_'+'rho='+str(param.rho)+'_'+str(param.repeat)+'.npz', self.hyb, self.hybmin, self.hybmax, self.nnm, self.nnmmin, self.nnmmax, self.all_e, self.allmin, self.allmax, self.phase2, self.p2min, self.p2max, self.d_axis)
import multiprocessing as mp
from functools import partial

def single_d( d, param, repeat ):
	
	print("d=",d)
	hybrid_X = 0
	nnm_X	 = 0
	sample = Samples(param, d)
	sample.sample_col(repeat)
	sample.sample_entry()
	recons 			= Reconstruct(param, sample, d)
	if param.indic.show_hybrid:
		hybrid_X 	= LA.norm(recons.hybrid_X-param.A.data,'fro')**2
	if param.indic.show_nnm:
		
		nnm_X 		= LA.norm(recons.nnm_X-param.A.data,'fro')**2
#		nnmcvx_X	= LA.norm(recons.nnmcvx_X-A.data,'fro')**2

	return hybrid_X, nnm_X

def iterate_d( d_axis, param ):
	norms_hyb 		= np.zeros(len(d_axis))
	norms_nnm 		= np.zeros(len(d_axis))
	hybrid_X_sum 	= 0
	nnm_X_sum	 	= 0
	
	hyb_list		= np.zeros(( len(d_axis), param.repeat ))
	nnm_list		= np.zeros(( len(d_axis), param.repeat ))
	hyb_max 		= np.zeros(len(d_axis))
	hyb_min 		= np.zeros(len(d_axis))
	nnm_max 		= np.zeros(len(d_axis))
	nnm_min 		= np.zeros(len(d_axis))

	for i in range(param.repeat):
		
#		print('repeat=',i)
		
		#multiprocessing
#		with mp.Pool() as p:
#			sequence = p.starmap(single_d, zip(d_axis,[param]*len(d_axis), [i]*len(d_axis)))
#		
#		d = 0
#		for hybrid_X, nnm_X in sequence:
#			
#			hyb_list[d,i] = hybrid_X
#			nnm_list[d,i] = nnm_X
#			norms_hyb[d] += hybrid_X
#			norms_nnm[d] += nnm_X
#			d += 1
		
		for d in range (len(d_axis)):
			
			hyb_list[d,i], nnm_list[d,i] = single_d(d_axis[d], param, i)
			norms_hyb[d] += hyb_list[d,i]
			norms_nnm[d] += nnm_list[d,i]

	norms_hyb /= param.repeat
	norms_nnm /= param.repeat
		
	for d in range (len(d_axis)):
		hybstd = np.std(hyb_list[d])
		nnmstd = np.std(nnm_list[d])
		hyb_max[d] = norms_hyb[d]+2*hybstd
		hyb_min[d] = norms_hyb[d]-2*hybstd
		nnm_max[d] = norms_nnm[d]+2*nnmstd
		nnm_min[d] = norms_nnm[d]-2*nnmstd
		
	
	return (norms_hyb, hyb_max, hyb_min, norms_nnm, nnm_max, nnm_min)
		