import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from helper import Matrix
import copy
import math
def compLevExact(A, k, axis):
	""" This function computes the column or row leverage scores of the input matrix.
	  
		:param A: n-by-d matrix
		:param k: rank parameter, k <= min(n,d)
		:param axis: 0: compute row leverage scores; 1: compute column leverage scores.
	
		:returns: 1D array of leverage scores. If axis = 0, the length of lev is n.  otherwise, the length of lev is d.
	"""

	U, D, V = np.linalg.svd(A, full_matrices=False)

	if axis == 0:
		lev = np.sum(U[:,:k]**2,axis=1)
	else:
		lev = np.sum(V[:k,:]**2,axis=0)

	p = lev/k

	return lev, p
def cur(A, k, c, r):
 
# This function computes the CUR decomposition of given matrix.
# Input:
#     A: n-by-d matrix.
#     k: a rank parameter.
#     c: number of columns to sample.
#     r: number of rows to sample.
# Output:
#     C,U,R: n-by-c, c-by-r, r-by-d matrices, respectively, such that ||A - CUR||_F is small.
#     lev_V, lev_U: leverage scores computed.

	# Computing leverage scores from A
	colLev, p = compLevExact(A, k, 1)
	bins = np.add.accumulate(p)
	colInd = np.digitize(np.random.random_sample(c), bins)

	# Sampling columns from A
	C = np.dot(A[:,colInd], np.diag(1/p[colInd]))

	# Computing leverage scores from C
	rowLev, p = compLevExact(C, c, 0)
	bins = np.add.accumulate(p)
	rowInd = np.digitize(np.random.random_sample(r), bins)

	# Sampling rows from A
	R = np.dot(np.diag(1/p[rowInd]),A[rowInd,:])
	W = np.dot(np.diag(1/p[rowInd]),C[rowInd,:])
	U = np.linalg.pinv(W)

	return C, U, R, rowLev, colLev	

def cur_norescale(A, k, c, r):
 
# This function computes the CUR decomposition of given matrix.
# Input:
#     A: n-by-d matrix.
#     k: a rank parameter.
#     c: number of columns to sample.
#     r: number of rows to sample.
# Output:
#     C,U,R: n-by-c, c-by-r, r-by-d matrices, respectively, such that ||A - CUR||_F is small.
#     lev_V, lev_U: leverage scores computed.

	# Computing leverage scores from A
	colLev, p = compLevExact(A, k, 1)
	bins = np.add.accumulate(p)
	colInd = np.digitize(np.random.random_sample(c), bins)

	# Sampling columns from A
	C = A[:,colInd]

	# Computing leverage scores from C
	rowLev, p = compLevExact(C, c, 0)
	bins = np.add.accumulate(p)
	rowInd = np.digitize(np.random.random_sample(r), bins)

	# Sampling rows from A
	R = A[rowInd,:]
	W = C[rowInd,:]
	U = np.linalg.pinv(W)

	return C, U, R, rowLev, colLev	

def cur_plus_solver( A_obs, U, V, omega_e, r):
	e_indices = tuple(zip(*omega_e))
	X = cp.Variable((r, r))
	R = U@X@V.T
	objective_fn = cp.norm(A_obs[e_indices]-R[e_indices], 2)
	problem = cp.Problem(cp.Minimize(objective_fn) )
	problem.solve()
	return X.value
	
	
def cur_plus(M, d1, d2, r):

	m,n = M.shape
	colInd = np.random.choice(n, d1, replace=False)
	rowInd = np.random.choice(m, d2, replace=False)
	# Sampling columns from M
	A = M[:,colInd]
	# Sampling rows from M
	B = M[rowInd,:]
	AAt = A@A.T
	BBt = B.T@B
	print("bbtshape",BBt.shape)
	_,Uh = LA.eig(AAt)
	_,Vh = LA.eig(BBt)
	print(Uh.shape, Vh.shape)
	Uh = Uh[:,:r]
	Vh = Vh[:,:r]
	print(Uh.shape, Vh.shape)
	
	entries = np.random.choice([1,0], (m,n))
	omega_e = []
	M_s = np.zeros((m,n))
	maxnum = d1*d2
	count = 0
	for i in range (m):
		for j in range (n):
			if entries[i,j]==1 and count<maxnum:
				omega_e.append((i,j))
				M_s[i,j] = M[i,j]
				count += 1
	print(omega_e)
	Z = cur_plus_solver( M_s, Uh, Vh, omega_e, r)
	return Uh, Z, Vh
	
def cur_plus_noise(M, d1, d2, r, sig_e2, sig_c2, maxnum):

	m,n = M.shape
	colInd = np.random.choice(n, d1, replace=False)
	rowInd = np.random.choice(m, d2, replace=False)
	# Sampling columns from M
	A = M[:,colInd]+np.random.normal(0,math.sqrt(sig_c2),(m,d1))
	# Sampling rows from M
	B = M[rowInd,:]+np.random.normal(0,math.sqrt(sig_c2),(d2,n))
	AAt = A@A.T
	BBt = B.T@B
#	print("bbtshape",BBt.shape)
	_,Uh = LA.eig(AAt)
	_,Vh = LA.eig(BBt)
#	print(Uh.shape, Vh.shape)
	Uh = Uh[:,:r]
	Vh = Vh[:,:r]
#	print(Uh.shape, Vh.shape)
	p = maxnum/m/n
	omega_e = []
	M_noise = np.zeros((m,n))
	count = 0
	for i in range (m):
		if count>=maxnum:
			break
		for j in range (n):
			q = np.random.uniform(0, 1)
			if q<=p:
				omega_e.append((i,j))
				M_noise[i,j] = M[i,j] + np.random.normal(0,math.sqrt(sig_e2))
				count += 1
	print(count,maxnum)
	Z = cur_plus_solver( M_noise, Uh, Vh, omega_e, r)
	return Uh, Z, Vh
	
if __name__ == "__main__":

#	k = 8
#	c = 6
#	r = 5
#
#	A = np.random.normal(5, 0.1, (20,16))
#	#A = A.T
#
#	U, D, V = np.linalg.svd(A, full_matrices=False)
#	Ak = np.dot( np.dot(U[:,:k], np.diag(D[:k])), V[:k,:])
#	baseline = np.linalg.norm(A - Ak, 'fro')
#	print ("||A-Ak||_F:",baseline)
#	C, U, R, rowLev, colLev = cur(A, k, c, r)
#	tmp = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
#	print ("rescaled||A-CUR||_F:",tmp, "||A-CUR||_F/||A||_F",tmp/np.linalg.norm(A, 'fro'))
#	
#	C, U, R, rowLev, colLev = cur_norescale(A, k, c, r)
#	tmp = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
#	print ("norescale||A-CUR||_F:",tmp, "||A-CUR||_F/||A||_F",tmp/np.linalg.norm(A, 'fro'))						
#	
#	U, Z, V = cur_plus(A, k, c, r)
#	tmp = np.linalg.norm(A - np.dot(np.dot(U,Z),V.T), 'fro')
#	print ("rand_entry||A-CUR||_F:",tmp, "||A-CUR||_F/||A||_F",tmp/np.linalg.norm(A, 'fro'))	

	noise = 2			#1 low
						#2 high
	if noise == 1:
		noisestr = "Low"
	else:
		noisestr = "High"
		
		
	choice = 2 			#1 = synthetic
						#2 = movielens
						#3 = jester
	if choice ==1:
		data = "synthetic"
	elif choice ==2:
		data = "movielens"
	else:
		data = "jester"

	if choice == 1:
		mu 		= 5
		sig2 	= 1
		m, n 	= 80, 60
		rank 	= 4
		targ_r 	= 3
		start 	= 3
		step	= 1
		repeat 	= 100
		A 		= np.random.normal(mu, math.sqrt(sig2), (m,n)) 
		
		U, D, V = LA.svd(A, full_matrices=False)
		A = np.dot( np.dot(U[:,:rank], np.diag(D[:rank])), V[:rank,:])
		
		U, D, V = LA.svd(A, full_matrices=False)
		Ak = np.dot( np.dot(U[:,:targ_r], np.diag(D[:targ_r])), V[:targ_r,:])
		baseline = np.linalg.norm(A - Ak, 'fro')
	#	A = Matrix(np.loadtxt('./data/movielens.txt', dtype=float).T)
	#	targ_r = 10
	#		
	#	A = Matrix(np.loadtxt('./data/jester.txt', dtype=float).T)
	#	targ_r = 10
		pe = 1
		pc = 0.2 * m
		budget = 3 * m * targ_r
		if noise == 1:
			sig_e2 = 0.1**2
		else:
			sig_e2 = 0.2**2
		gamma = 5
		sig_c2 = 5*sig_e2
		
		maxcol = math.floor(budget/2/pc)
		repeat = 10
		
	if choice == 2:

		m, n 	= 1682, 983
		targ_r 	= 10
		start 	= 120
		step 	= 20
		repeat 	= 5
		A = np.loadtxt('./data/movielens.txt', dtype=float).T
		
		U, D, V = LA.svd(A, full_matrices=False)
		Ak = np.dot( np.dot(U[:,:targ_r], np.diag(D[:targ_r])), V[:targ_r,:])
		baseline = np.linalg.norm(A - Ak, 'fro')
	#	A = Matrix(np.loadtxt('./data/movielens.txt', dtype=float).T)
	#	targ_r = 10
	#		
	#	A = Matrix(np.loadtxt('./data/jester.txt', dtype=float).T)
	#	targ_r = 10
		pe = 1
		pc = 0.2 * m
		budget = 10 * m * targ_r
		if noise == 1:
			sig_e2 = 0.05**2
		else:
			sig_e2 = 0.15**2
		gamma = 5
		sig_c2 = gamma*sig_e2
		
		maxcol = math.floor(budget/2/pc)
	
	if choice == 3:

		m, n 	= 7200,100
		targ_r 	= 10
		start 	= 10
		step 	= 1
		repeat 	= 5
		A = np.loadtxt('./data/jester.txt', dtype=float).T
		
		U, D, V = LA.svd(A, full_matrices=False)
		Ak = np.dot( np.dot(U[:,:targ_r], np.diag(D[:targ_r])), V[:targ_r,:])
		baseline = np.linalg.norm(A - Ak, 'fro')
	#	A = Matrix(np.loadtxt('./data/movielens.txt', dtype=float).T)
	#	targ_r = 10
	#		
	#	A = Matrix(np.loadtxt('./data/jester.txt', dtype=float).T)
	#	targ_r = 10
		pe = 1
		pc = 0.2 * m
		budget = 1.1 * m * targ_r
		if noise == 1:
			sig_e2 = 0.2**2
		else:
			sig_e2 = 0.5**2
		gamma = 10
		sig_c2 = gamma*sig_e2
		
		maxcol = math.floor(budget/2/pc)
	
	daxis = []
	curmin = []
	curmax = []
	curavg = []
	for i in range(start, maxcol, step):
		maxnum = budget-2*i*pc
		daxis.append(i)
		
		tmp = []
		for j in range(repeat):
		
			U, Z, V = cur_plus_noise(A, i, i, targ_r, sig_e2, sig_c2, maxnum)
			tmp.append(LA.norm(A - np.dot(np.dot(U,Z),V.T), 'fro'))
		
		cur_std = np.std(tmp)
		cur_avg = np.mean(tmp)
		
		curmin.append(cur_avg-2*cur_std)
		curmax.append(cur_avg+2*cur_std)
		curavg.append(cur_avg)
	print(curavg, curmax, curmin)
	np.savez('./numpy/'+data+'_'+noisestr+'.npz', daxis, curavg, curmin, curmax)