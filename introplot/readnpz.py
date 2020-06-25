import os, sys 
import copy
import numpy as np
from numpy import linalg as LA

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import matplotlib.ticker as mtick
rcParams['font.family'] = 'serif'
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams['font.size'] = 10
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D
from labellines import labelLine, labelLines

a=np.load("arrays1.npz")
b=np.load("arrays2.npz")
c=np.load("arrays3.npz")
d=np.load("arrays4.npz")

hyb_low		= b['hyb']
nnm_e_low 	= a['nnm_e']
nnm_2p_low	= a['nnm_2p']

hyb_high	= c['hyb']
nnm_e_high 	= d['nnm_e']
nnm_2p_high	= d['nnm_2p']

cur_low = np.array([26.7, 24.1, 22.6, 21.5, 21, 19.9, 20.1, 18.5, 18.4, 18.3])
betas = d['betas']*100

nrlogn = 140*4*math.log10(140)/4800*100
print(nrlogn)


start = 1
end = 10
stderr = 223.90151411973596
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_xlabel(r'$\frac{ \#\ \mathrm{of\ i.i.d\ sampled\ entries}}{ \#\ \mathrm{of\ total\ entries}}$',fontsize=15)
ax.set_ylabel(r'$\frac{ \Vert \overline{\mathbf{A}}-\mathbf{A} \Vert^2_F}{ \Vert \mathbf{A}_r-\mathbf{A} \Vert^2_F}$', fontsize=15)

ax.plot(betas[start:end], cur_low[start:end]**2/stderr, c='orange',label='CUR+')
#ax.set_yscale('log')
ax.plot(betas[start:end], hyb_low[start:end]/stderr, c='b',label='nCUR(This paper)')
#ax.plot(betas[start:end], hyb_high[start:end]/stderr,  c='b',label='HReg(This paper)')
ax.plot(betas[start:end], nnm_e_low[start:end]/stderr, c='purple',label='NNa')
#ax.plot(betas[start:end], nnm_e_high[start:end]/stderr,  c='purple',label='NNa')
ax.plot(betas[start:end], nnm_2p_low[start:end]/stderr, c='olive',label='Chen')
#ax.plot(betas[start:end], nnm_2p_high[start:end]/stderr,  c='yellow',label='Chen')
ax.axvline(x=nrlogn, linestyle='-.', label='nrlog^2', c='black')
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
#ax.legend(loc='upper right')
ax.annotate(r'$\frac{(n+m)rlog(n+m)}{mn}$', xy=(1190, 890), xycoords='figure pixels')
labelLines(plt.gca().get_lines(),zorder=2.5)
ax.axvspan(betas[start], nrlogn, alpha=0.1, color='red')
ax.axvspan(nrlogn, betas[end-1], alpha=0.1, color='green')
plt.savefig('compare.png',dpi=300,bbox_inches='tight')
plt.show()