import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals.six.moves import zip
from sklearn import svm
from sklearn.svm import SVC
from sklearn
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, SGDClassifier
from scipy.stats import itemfreq
from sklearn.externals import joblib
import pickle
from topbottom_cluster import consolidate_vars
from clustering import r_value


class market:
	def __init__(self,categorical=True,norm=None,random=False):
		self.version = version
		self.categorical = categorical
		if categorical == True:
			self.prob = True
		else:
			self.prob = False
		if norm == 'std' or norm == 'maxmin':
			self.norm = norm
			self.norm_type = True
		elif norm == None:
			self.norm_type = False
		else: 
			print ("norm must be 'std','maxmin', or None")
		self.random = random
		
		
	def sample(self,X,y,div):
		if self.random==True:
			X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=div,random_state=0)
		else:
			n=len(X)
			X_train=X[div:n]
			y_train=y[div:n]
			X_test=X[0:div]
			y_test=y[0:div]
		return X_train,y_train,X_test,y_test


	def tune(self,X_train,y_train,method):
		model={}
		if method == 'SVM':
			#parameters to tune model
			tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-1,1e-3,1e-5],'C':[1,10,100]},
					{'kernel':['linear'],'C':[1,100],'gamma':[1e-3]}]
			#use gridsearch to find best model
			clf = GridSearchCV(SVC(probability=self.prob),tuned_parameters)
			clf.fit(X_train,y_train)
			print(clf.best_estimator_)
			model['svm']=clf
			# if self.prob == True:
			# 	sgd = SGDClassifier(loss='log',n_iter=100,alpha=.01,class_weight="auto")
			# else:
			# 	sgd = SGDClassifier(n_iter=100,alpha=.01,class_weight="auto")
			# sgd.fit(X_train,y_train)
			# model['sgd'] = sgd
		elif method == 'NB':
			gnb=GaussianNB()
			gnb.fit(X_train,y_train)
			model['gnb'] = gnb
		elif method == 'LN':
			ln= LogisticRegression()
			model['log'] = ln
		elif method == 'ADT':
			bdt = {}
			bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
				n_estimators=600,
				learning_rate=1.5,
				algorithm="SAMME")
			bdt_discrete.fit(X_train,y_train)
			# bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
			# 	n_estimators=600,
			# 	learning_rate=1)
			# bdt_real.fit(X_train,y_train)
			model['bdt_discrete'] = bdt_discrete
			# model['bdt_real'] = bdt_real
		elif method == 'poly':
			poly_model={}
			for degree in [1,2]:
				print('degree '+str(degree)+'...')
				poly=make_pipeline(PolynomialFeatures(degree),Ridge(normalize=self.norm_type))
				if self.version == 'high' or self.version == 'low':
					poly.fit(X_train[:,1:],y_train)
				else: 
					poly.fit(X_train,y_train)
				model['poly-'+str(degree)] = poly
		elif method == 'linear':
			br = BayesianRidge(compute_score=True,normalize=self.norm_type)
			br.fit(X_train,y_train)
			model['Bayesian_Ridge'] = br
			ols = LinearRegression(normalize=self.norm_type)
			ols.fit(X_train,y_train)
			model['OLS'] = ols
		return model

	def pred(self,model,X_test,y_test):
		d={}
		d['y_test']=y_test
		for m in model:
			mod=model[m]
			if self.version == 'high' or self.version == 'low':
				d[m] = mod.predict(X_test[:,1:])
			else:
				d[m]=mod.predict(X_test)
			if self.prob == True:
				probas = mod.predict_proba(X_test)
				values = np.unique(d[m])
				n = np.shape(probas)[1]
				for i in range(n):
					prob_name = str(m)+'_prob_'+str(i)
					d[prob_name] = probas[:,i]
		return d

	def main(self,data,method,perc,smp=1):
		d={}
		if smp>0:
			n=int(len(data))
			ns=n/float(smp)
			start=0
			div=int(round(ns*perc))
			for i in range(smp):
				end=int(round(start+ns))
				y=np.array(data.ix[start:end,0])
				X=np.matrix(data.ix[start:end,1:])
				y=y.astype(float)
				if self.norm_type == True:
					if self.norm == 'std':
						scaler = preprocessing.StandardScaler().fit(X)
					elif self.norm == 'maxmin':
						scaler = preprocessing.MinMaxScaler().fit(X)
					X = scaler.transform(X)
				X = X.astype(float)
				X_train,y_train,X_test,y_test=self.sample(X,y,div)
				model=self.tune(X_train,y_train,method)
				for m in model:
					model_name = './models/'+str(m)+'.pkl'
					joblib.dump(model[m],model_name)
				if perc == 0:
					d_temp = self.pred(model,X_train,y_train)
				else:
					d_temp=self.pred(model,X_test,y_test)
				for key in d_temp:
					try:
						d[key] = np.append(d[key],d_temp[key])
					except KeyError:
						d[key] = d_temp[key]
				start=int(start+div)
			for key in d:
				if 'prob' not in key and key!='y_test':
					if self.categorical == True:
						print('y_true\n'+str(itemfreq(d['y_test'])))
						print(str(key)+'\n'+str(itemfreq(d[key])))
						print('accuracy: '+str(accuracy_score(d['y_test'],d[key])))
					else:
						m = np.shape(X_test)[1]-1
						print(str(key)+': '+str(r_value(d['y_test'],d[key],m)))
			Y=pd.DataFrame(d)
		else:
			print ('SMP must be greater than 0.')

		return X_test, y_test, model, Y

	def read_data(self,__file__,method,perc,smp=1):
		data=pd.read_csv(__file__)
		X_test, y_test, model, Y = self.main(data,method,perc,smp)
		name=str(__file__.replace('.csv','_'))+str(method)+'_pred.csv'
		print ('output labeled as '+str(name))
		Y.to_csv(name,sep=',',index=False)

if __name__=='__main__':
	__file__='legal_docs.csv'
	MRK = market(categorical=True,norm='std',random=True)
	MRK.read_data(__file__,'SVM',.2)




