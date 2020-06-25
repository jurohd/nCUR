'''This is the main experiment file'''

import os, sys 
import copy
import numpy as np
from numpy import linalg as LA

import math

from helper import Indicator, Matrix, Params, iterate_d, Plot_params
#from matrixgen import matrix_generate
from algorithm import Samples, Reconstruct, Hybrid_recover, Nnm_recover
from algorithm import All_entry_nnm, Twophase_nnm
#from reconstruction import nnm_reconstruct, hybrid_reconstruct

dataTYPE 	= sys.argv[1]
HIGH_BUD 	= sys.argv[2]
HIGH_NOI 	= sys.argv[3]


if HIGH_BUD == "T":
	bud_indic = True
else:
	bud_indic = False

if HIGH_NOI == "T":
	noi_indic = True
else:
	noi_indic = False

method_indicator = Indicator(	data_type 	= dataTYPE,
 								show_hybrid = True,
								show_nnm 	= False, 		#normal nnm method
								show_nnm_e	= False,			#all entry nnm 
								show_2phase = False,			#two phase sampling 
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

if method_indicator.data_type == "movielens":
	A = Matrix(np.loadtxt('./data/movielens.txt', dtype=float).T)
	targ_r = 10

if method_indicator.data_type == "jester":
	A = Matrix(np.loadtxt('./data/jester.txt', dtype=float).T)
	targ_r = 10
	

param = Params(	indic			= method_indicator,
				is_synthetic 	= method_indicator.data_type, 
				matrix_info 	= A, 
				h_budget 		= method_indicator.h_budget, 
				h_noise 		= method_indicator.h_noise, 
				targ_r 			= targ_r, 
				alpha 			= 0.2
			)	
#param.lbd = float(sys.argv[4])
param.rho = float(sys.argv[4])
param.repeat = int(sys.argv[5])
print('rho is',param.rho, 'rpt=',param.repeat)

d_axis = np.arange(int(param.lplotcoeff*param.targ_r), int(param.maxnum_cols-param.rplotcoeff*targ_r),	param.d_interval )
plts = Plot_params(d_axis)

if param.indic.show_nnm_e:
	plts.set_all_entry(All_entry_nnm(param, plts.length))
		
if param.indic.show_2phase:
	plts.set_phase2(Twophase_nnm(param, plts.length)) 

#param.lbd = lbd
if param.indic.show_hybrid:
	plts.set_hyb_nnm(iterate_d( d_axis, param )) 
#print(param.lbd)
plts.plotgraph(param, method_indicator)