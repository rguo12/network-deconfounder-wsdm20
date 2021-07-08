#!/usr/bin/env python
# coding: utf-8

# In[64]:
import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc, rcParams
#visualize everything using tsne
from tsne import tsne as tsn
from tsne import pylab

from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse

font = {'weight' : 'bold',
        'size'   : 14}

rc('font', **font)
path = './datasets/'
name = 'Flickr'
data = sio.loadmat(path+name+'/data.mat')


# In[76]:


#setting constants
kappa1 = 10
kappa2 = 10
C = 1
att = "something"

if kappa2 != 0.1:
	extra_str = str(kappa2)
else:
	extra_str = ''

if att!="":
	extra_str += 's'



# In[3]:


data.keys()


# In[126]:


X = data['Attributes']
n=X.shape[0]
X.tocoo()


# In[13]:


A = data['Network']
A_dense = np.array(A.todense())

# feel free to change this in (0,1], the larger the similar to an unweighted network 
LOW = 0.1

if att!="":
	#randomly sample weights for the Adjacency matrix
	for i in range(n):
		for j in range(i+1,n):
			if A_dense[i,j]!=0:
				A_dense[i,j]=np.random.uniform(low=LOW, high=1.0, size=1)
				A_dense[j,i]=A_dense[i,j]

Reweighted_Sparse_A = sparse.csr_matrix(A_dense)

# In[6]:


#check if A is symmetric
# def check_symmetric(a, tol=1e-8):
#     return np.allclose(a, a.T, atol=tol)
# check_symmetric(A_dense)


# In[7]:


#get 50 topics
lda = LatentDirichletAllocation(n_components=50)
lda.fit(X)


# In[8]:


Z = lda.transform(X)


# In[15]:


AZ = np.matmul(A_dense,Z)


# In[18]:


# AZ.shape


# In[25]:


#random sample an instance and use its topic distribution as the centroid
for exp_id in range(0,10):
	centroid1_idx = random.randint(0, X.shape[0]-1)
	Z_c1 = Z[centroid1_idx,:]


	# In[26]:


	Z_c0 = np.mean(Z,axis=0)


	# In[35]:


	#precompute the similarity between each instance and the two centroids
	ZZ_c1 = np.matmul(Z,Z_c1)
	ZZ_c0 = np.matmul(Z,Z_c0)
	AZZ_c1 = np.matmul(AZ,Z_c1)
	AZZ_c0 = np.matmul(AZ,Z_c0)


	# In[77]:


	#get propensity for each instance
	p1 = kappa1*ZZ_c1+kappa2*AZZ_c1
	p0 = kappa1*ZZ_c0+kappa2*AZZ_c0
	propensity = np.divide(np.exp(p1), np.exp(p1)+np.exp(p0))


	# In[78]:


	ps = pd.Series(np.squeeze(propensity))
	ps.describe()


	# In[113]:


	#visualize the propensity distribution
	# %matplotlib inline
	fig0, ax0 = plt.subplots()
	ax0.hist(propensity,bins=50)
	plt.title('propensity score distribution')
	plt.xlabel('propensity score')
	plt.ylabel('frequency')
	plt.savefig('./figs/'+name+extra_str+str(exp_id)+'ps_dist.pdf',bbox_inches='tight')
	# plt.show()


	# In[80]:


	#simulate treatments
	T = np.random.binomial(1, p=propensity)


	# In[81]:


	# plt.hist(T,bins=50)
	# plt.title('Treatment')
	# plt.savefig('./figs/'+name+'ps_dist.pdf',bbox_inches='tight')
	# plt.show()


	# In[82]:


	#sample noise from Gaussian
	epsilon = np.random.normal(0,1,X.shape[0])

	# In[83]:

	#simulate outcomes
	Y1 = C*(p1+p0)+epsilon
	Y0 = C*(p0)+epsilon

	# In[112]:

	fig1, ax1 = plt.subplots()
	ax1.hist(Y1,bins=50,label='Treated')
	ax1.hist(Y0,bins=50,label='Control')
	plt.title('outcome distribution')
	plt.legend()
	plt.savefig('./figs/'+name+extra_str+str(exp_id)+'outcome_dist.pdf',bbox_inches='tight')
	# ax1.show()


	# In[111]:


	#distribution of ITE
	fig2, ax2 = plt.subplots()
	ax2.hist(Y1-Y0,bins=50,label='ITE')
	plt.title('ITE distribution')
	plt.xlabel('ITE')
	plt.ylabel('frequency')
	ax2.axvline(x=np.mean(Y1-Y0),color='red',label='ATE')
	plt.savefig('./figs/'+name+extra_str+str(exp_id)+'ite_dist.pdf',bbox_inches='tight')
	ax2.legend()
	# plt.show()


	# In[86]:


	print('ATE is %.3f'%(np.mean(Y1-Y0)))


	# In[87]:


	#save the data
	#save Y1 Y0 T X


	# In[88]:


	# Z_ = tsn(Z, 2, 50, 20.0)


	# In[110]:


	# labels = T #use treatment as the binary label
	# treated_idx = np.where(T==1)[0]
	# controled_idx = np.where(T==0)[0]
	# fig3, ax3 = plt.subplots()
	# ax3.scatter(Z_[treated_idx, 0], Z_[treated_idx, 1], 3,marker='o',color='red')
	# ax3.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3,marker='o',color='blue')
	# # ax1.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3,marker='o',color='yellow')
	# ax3.scatter(np.mean(Z_[:,0]),np.mean(Z_[:,1]),100,label=r'$z_0^c$',marker='D',color='yellow')
	# ax3.scatter(Z_[centroid1_idx,0],Z_[centroid1_idx,1],100,label=r'$z_1^c$',marker='D',color='green')
	# # fig2, ax2 = plt.subplots()

	# # ax2.scatter(np.mean(Z_[:,0]),np.mean(Z_[:,1]),100,label='centroid_0',marker='o',color='black')
	# # ax2.scatter(Z_[centroid1_idx,0],Z_[centroid1_idx,1],100,label='centroid_1',marker='o',color='blue')
	# plt.savefig('./figs/'+name+extra_str+'tsne.pdf',bbox_inches='tight')
	# plt.legend(loc=2)
	# plt.xlim(-100,100)
	# pylab.show()


	# In[114]:


	#get the most freq 100 words of each topic
	topics = lda.components_


	# In[115]:


	#calculate the topic 100 words in each topic
	topics_100_dims = np.argsort(topics,axis=1)[:,-100:]


	# In[116]:


	#then we get a union of all those top 100 words
	unique_100_dims = np.unique(topics_100_dims)


	# In[117]:


	#reduce the dimensions by extract the selected words
	X_100 = X[:,unique_100_dims]


	# In[118]:


	# X_100.shape


	# In[122]:


	#save the data
	sio.savemat('./datasets/'+name+extra_str+'/'+name+str(exp_id)+'.mat',{
	    'X_100':X_100, 'T':T, 'Y1':Y1, 'Y0':Y0, 'Attributes': data['Attributes'], 'Label':data['Label'],
	    'Network':data['Network'], 'Weighted_Network':Reweighted_Sparse_A})


	# In[ ]:




