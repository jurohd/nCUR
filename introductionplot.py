import os, sys 
import copy
import numpy as np
from numpy import linalg as LA

import math

from helper import Indicator, Matrix, Params, iterate_d, Plot_params
#from matrixgen import matrix_generate
from algorithm import Samples, Reconstruct, Hybrid_recover, Nnm_recover
from algorithm import All_entry_nnm, Twophase_nnm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D

dataTYPE 	= "synthetic"
HIGH_BUD 	= "T"
HIGH_NOI 	= "T"


if HIGH_BUD == "T":
	bud_indic = True
else:
	bud_indic = False

if HIGH_NOI == "F":
	noi_indic = True
else:
	noi_indic = False

method_indicator = Indicator(	data_type 	= dataTYPE,
 								show_hybrid = True,
								show_nnm 	= False, 		#normal nnm method
								show_nnm_e	= True,			#all entry nnm 
								show_2phase = True,			#two phase sampling 
								h_budget 	= bud_indic, 	#budget parameter
								h_noise 	= noi_indic	
							) 

if method_indicator.data_type == "synthetic":
	mu 		= 5
	sig2 	= 1
	m, n 	= 80, 60
	targ_r 	= 3
	A 		= Matrix( np.random.normal(mu, math.sqrt(sig2), (m,n)) )
	A.rank_r_proj(4) #trunc the rank to be 4


param = Params(	indic			= method_indicator,
				is_synthetic 	= method_indicator.data_type, 
				matrix_info 	= A, 
				h_budget 		= method_indicator.h_budget, 
				h_noise 		= method_indicator.h_noise, 
				targ_r 			= targ_r, 
				alpha 			= 0.2
			)	

betas = np.linspace(0.06,1,30,endpoint=True)
lbds = np.logspace(-5,1,1)
print(lbds)
hyb 	= []
nnm_e 	= []
nnm_2p	= []
count=0
#print(betas)
for beta in betas:
	param.budget = int(m*n*beta)
	param.print_budget()
	#print('rho is',param.rho, 'rpt=',param.repeat)

	d_axis = np.arange(5, int(param.maxnum_cols-param.rplotcoeff*targ_r),	param.d_interval )
	plts = Plot_params(d_axis)

	if param.indic.show_nnm_e:
		plts.set_all_entry(All_entry_nnm(param, plts.length))
		
	if param.indic.show_2phase:
		plts.set_phase2(Twophase_nnm(param, plts.length)) 

	minhyb = np.inf
	for lbd in lbds:
		print(lbd)
		param.lbd = lbd
		if param.indic.show_hybrid:
			plts.set_hyb_nnm(iterate_d( d_axis, param )) 
			cur_min = np.min(plts.hyb)
			if cur_min<minhyb:
				print("subst")
				minhyb=cur_min
	#print(param.lbd)
#	plts.plotgraph(param, method_indicator)
	if len(plts.hyb) != 0:
		hyb.append(minhyb)
		nnm_e.append(np.min(plts.all_e))
		nnm_2p.append(np.min(plts.phase2))
	else:
		count+=1

for i in range(len(hyb)):
	print(hyb[i],nnm_e[i],nnm_2p[i])
betas = betas[count:]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_xlabel('budget(percentage of matrix that can be sampled entry-wisely)')
ax.set_ylabel('relative error to rank r approximation')
#ax.set_yscale('log')
ax.plot(betas, hyb, c='b',label='Hyb-Reg(Our Method)')
ax.plot(betas, nnm_e, c='purple',label='NNM-all_entry(Candes)')
ax.plot(betas, nnm_2p, c='yellow',label='NNM-2phase(Chen)')
ax.legend(loc='upper right')
plt.show()
np.savez('./introplot/arrays.npz',betas=betas,hyb=hyb,nnm_e=nnm_e,nnm_2p=nnm_2p)

