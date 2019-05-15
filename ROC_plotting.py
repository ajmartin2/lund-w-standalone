
# coding: utf-8

# In[1]:


import matplotlib.pyplot as pl
import matplotlib.colors as colors
import matplotlib.cm as cmap
import numpy as np


# In[2]:


roc_2vars=np.load('roc_W_lowpt_2var.npy')
roc_3vars=np.load('roc_W_lowpt_3var.npy')
fpr_tops_2vars=np.load('fpr_W_lowpt_2var.npy')
fpr_tops_3vars=np.load('fpr_W_lowpt_3var.npy')
tpr_tops_2vars=np.load('tpr_W_lowpt_2var.npy')
tpr_tops_3vars=np.load('tpr_W_lowpt_3var.npy')


# In[3]:


pt_min=200
pt_max=500


# In[4]:


pl.figure()
lw = 2
fpr_tops_2vars[fpr_tops_2vars==0.]=1e-5
fpr_tops_3vars[fpr_tops_3vars==0.]=1e-5
#print fpr[0]
pl.plot(tpr_tops_2vars, 1./fpr_tops_2vars, color='navy',lw=lw, label='$\ln(1/\Delta),\ln(p_T\cdot\Delta)$ (area = %0.2f)' % roc_2vars[0])
pl.plot(tpr_tops_3vars, 1./fpr_tops_3vars, color='orange',lw=lw, label='$\ln(1/\Delta),\ln(p_T\cdot\Delta),m$ (area = %0.2f)' % roc_3vars[0])
pl.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
pl.xlim([0.3, 1.0])
#pl.ylim([0.0, 1.05])
pl.yscale('log')
pl.ylim(1,1e4)
pl.xlabel('Signal efficiency ($\epsilon_{sig}$)')
pl.ylabel('Background rejection ($1/\epsilon_{bkg}$)')
#pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
ptrange='pT = [{0}, {1}] GeV'.format(int(pt_min),int(pt_max))
pl.text(0.75,4e3,ptrange)
# pl.show()
pl.savefig('Combined_CNN_ROC.pdf')
pl.close()
