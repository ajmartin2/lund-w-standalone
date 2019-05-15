
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
import numpy as np


# In[2]:


import numpy as np
import h5py
import matplotlib.pyplot as pl
import matplotlib.colors as colors
import matplotlib.cm as cmap


# In[3]:


# Open bkg and signal files
f_bkg = h5py.File('mcdata/mc16_13TeV:mc16_13TeV.361024.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4W.deriv.DAOD_JETM6.e3668_s3126_r9364_r9315_p3600.h5')
f_sig = h5py.File('mcdata/signal_Vs.h5')
#f_sig = h5py.File('Zprimett.merge.h5')


# In[4]:


arr_sig=f_sig['lundjets_InDetTrackParticles']
arr_bkg=f_bkg['lundjets_InDetTrackParticles']


# In[5]:


# Jet pt
pT_bkg = np.array(arr_bkg[:,:,:,2]).flatten()
print(pT_bkg[:10])
pT_sig = np.array(arr_sig[:,:,:,2]).flatten()
print(pT_sig[:10])


# In[6]:


x_bkg = np.log(1/arr_bkg[:,:,:,0]).flatten()
y_bkg = np.log(np.array(arr_bkg[:,:,:,0]).flatten()*np.array(arr_bkg[:,:,:,5]).flatten())
x_sig = np.log(1/arr_sig[:,:,:,0]).flatten()
y_sig = np.log(np.array(arr_sig[:,:,:,0]).flatten()*np.array(arr_sig[:,:,:,5]).flatten())


# In[7]:


# Plot Lund plane (no pt cut)
pl.hist2d(x_bkg, y_bkg, bins=40, range=[[0,5],[-2,7]], norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
pl.show()
pl.hist2d(x_sig, y_sig, bins=40, range=[[0,5],[-2,7]], norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
# pl.show()
pl.savefig('nocut_lundplane.pdf')
pl.close()
print(pT_bkg.shape)


# In[9]:


# Plot Lund plane (Jet pt > 1500)
pl.hist2d(x_bkg[pT_bkg > 1500], y_bkg[pT_bkg > 1500], bins=40, range=[[0,5],[-2,7]], norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
pl.show()
pl.hist2d(x_sig[pT_sig > 1500], y_sig[pT_sig > 1500], bins=40, range=[[0,5],[-2,7]], norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
# pl.show()
pl.savefig('1500cut_lundplane.pdf')
pl.close()
print(pT_bkg[pT_bkg > 1500].shape)
print(pT_bkg[pT_sig > 1500].shape)


# In[10]:


# Signal Lund plane (jet pt > 1000)
pl.hist2d(x_sig[pT_sig > 1000], y_sig[pT_sig > 1000], bins=40, range=[[0,5],[-2,7]], norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
# pl.show()
pl.savefig('1000cut_lundplane.pdf')
pl.close()
print(pT_bkg[pT_sig > 1000].shape)


# In[11]:


# Lund plane (jet pt > 500)
pl.hist2d(x_bkg[pT_bkg > 500], y_bkg[pT_bkg > 500], bins=40, norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
# pl.show()
pl.savefig('500cut_bkgplane.pdf')
pl.close()
pl.hist2d(x_sig[pT_sig > 500], y_sig[pT_sig > 500], bins=40, norm=colors.LogNorm(), cmap=cmap.jet)
pl.colorbar()
# pl.show()
pl.savefig('500cut_sigplane.pdf')
print(pT_bkg[pT_bkg > 500].shape)
print(pT_bkg[pT_sig > 500].shape)

