import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
sns.set_style("white")
random.seed(0)


class PUWrapper(object):
	def __init__(self,clf,n_fold=5):
		self.clf = clf
		self.n_fold = n_fold

	def fit(self,X,s):
		# estimate p(s=1|x) to estimate p(s=1|y=1)
		self.clf.fit(X,s) # claasify labeled or not

		Xp = X[s==1] # labeled
		n = len(Xp) # number of labeled
		
		cv_split = np.arange(n) * self.n_fold/n
		cv_index = cv_split[random.permutation(n)] # shuffle
		cs = np.zeros(self.n_fold)
		for k in range(self.n_fold):
			Xptr = Xp[cv_index==k]
			cs[k] = np.mean(self.clf.predict_proba(Xptr)[:,1])
		self.c_ = cs.mean()
		return self

	def predict_proba(self,X):
		proba=self.clf.predict_proba(X) / self.c_
		# p(y=1|x) = p(s=1|x)/c
		# where c = p(s=1|y=1) = mean(p(s=1|x))
		return proba

	def predict(self,X):
		proba=self.predict_proba(X)[:,1]
		return proba >= 0.5 #proba>=(0.5*self.c_)


if __name__ == "__main__":
	n1 = 100
	n2 = 500
	n = n1 + n2
	mean1 = [0, 0]
	mean2 = [2, 2]
	cov1 = 0.1*np.eye(2)
	cov2 = 0.5*np.eye(2)

	X = np.concatenate([np.random.multivariate_normal(mean1, cov1, n1),
						np.random.multivariate_normal(mean2, cov2, n2)])
	y = np.concatenate([np.ones(n1), np.zeros(n2)]) # label
	s = np.zeros(n, dtype=np.int32) # labeled or not

	index_labeled = np.arange(n)[y==1][:np.int32(n1*0.3)]
	s[index_labeled] = 1

	score = lambda l1, l2: metrics.f1_score(l1, l2, average=None)[1]
	scorer = metrics.make_scorer(score)
	
	clf = PUWrapper(LogisticRegression()).fit(X, s)
	
	print("accuracy (PU):"\
		  ,metrics.accuracy_score(y[s==0],clf.predict(X[s==0])))
	print("pos's F1 (PU):",score(y[s==0],clf.predict(X[s==0])))
	print("p(s=1|y=1): ", clf.c_)

	
	offset=.5
	xx,yy=np.meshgrid(np.linspace(X[:,0].min()-offset,X[:,0].max()+offset,100),
					  np.linspace(X[:,1].min()-offset,X[:,1].max()+offset,100))
	label2=clf.predict(X)
	proba=clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])
	Z2=proba[:,1]
	Z2=Z2.reshape(xx.shape)

	a1=plt.contour(xx, yy, Z2, levels=[0.5], linewidths=2, colors='green')
	b1=plt.scatter(X[label2==1][:,0],X[label2==1][:,1],c="blue",s=50)
	b2=plt.scatter(X[label2==0][:,0],X[label2==0][:,1],c="red",s=50)
	plt.axis("tight")
	plt.xlim((X[:,0].min()-offset,X[:,0].max()+offset))
	plt.ylim((X[:,1].min()-offset,X[:,1].max()+offset))
	plt.legend([a1.collections[0],b1,b2],
			   ["decision boundary","positive","negative"],
			   prop={"size":10},loc="upper left")
	plt.title("Result of PU classification")
	plt.tight_layout()
	plt.show()
