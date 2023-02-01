import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.neighbors import KNeighborsTimeSeries
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
from tslearn.svm import TimeSeriesSVC
from tslearn.preprocessing import TimeSeriesScalerMinMax
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import tsaug


class Sampler:
    def __init__(self, sampler, strg):
        self.sampler = sampler
        self.strg = strg
        self.fSampler = None
        
    def __ros__(self):
        self.fSampler = RandomOverSampler(sampling_strategy=self.strg)
    
    def __jittering__(self, x, sigma=0.03):
        return tsaug.AddNoise(scale = sigma).augment(x)
    
    def __timeWarping__(self, x, speed_change = 3, max_speed_rat = 2 ):
        """
        The augmenter random changed the speed of timeline. The time warping is controlled by the number of speed changes and the maximal ratio of max/min speed.

        Args:
            x (_type_): _description_
            speed_change (int, optional): _description_. Defaults to 3.
            max_speed_rat (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        return tsaug.TimeWarp(n_speed_change = speed_change, max_speed_ratio = max_speed_rat).augment(x)
    
    
    
    def __smote__(self, k_neig = 2):
        
        self.fSampler = SMOTE(sampling_strategy=self.strg,k_neighbors = k_neig)
        
        
    def __adasyn__(self, k_neig = 2):
        self.fSampler = ADASYN(sampling_strategy= self.strg, n_neighbors=k_neig)
        
        
        
        
    def sampleData(self, x_e,y_e):
        """Return sampled data

        Args:
            x (list of list): list containing data [data1, data2, data3, ...]

        Returns:
            _type_: _description_
        """
        x = list(x_e.copy())
        y = list(y_e.copy())
        
        _, y_dist = np.unique(y, return_counts=True)
        missing_to_add = dict()
        
        for i in range(len(y_dist)):
            missing_to_add[i] = self.strg[i] - y_dist[i]
            
        
        if (self.sampler == 'ROS'):
            return RandomOverSampler(sampling_strategy=self.strg).fit_resample(x,y)
        
        elif (self.sampler == 'Jitter'):
            for i in range(len(missing_to_add)):
                for j in range(missing_to_add[i] + 1):
                    if (j != 0):
                        
                        selected = x[np.random.choice(np.where(np.array(y) == i)[0])] # Select one random data with the label i
                        selected = tsaug.AddNoise(scale = 0.3).augment(selected)
                        x.append(selected)
                        y.append(i)
            return x, y

        elif (self.sampler == 'TW'):
            for i in range(len(missing_to_add)):
                for j in range(missing_to_add[i] + 1):
                    if (j != 0):
                        print(i,j)
                        
                        selected = x[np.random.choice(np.where(np.array(y) == i)[0])] # Select one random data with the label i
                        
                        selected = tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=2).augment(selected)
                        x.append(selected)
                        y.append(i)
            return x, y
        
        elif(self.sampler == 'SMOTE'):
            return SMOTE(k_neighbors=2, sampling_strategy=self.strg).fit_resample(x,y)
        
        elif(self.sampler == 'ADASYN'):
            return ADASYN(N_neighbors=2, sampling_strategy=self.strg).fit_resample(x,y)
            
            
            
            
            
        

class Classif:
    def __init__(self, clf):
        self.clf = clf  
        
              
        
    def fit(self, x_train, y_train):
        """
        Fit the model on training data
        
        """
        self.clf.fit(self,x_train, y_train)
        print('=== Model Fitted ===')
    
    def predict(self,x_test):
        """
        Realise prediction on input

        Args:
            x_test (list): test data
        """
        return self.clf.predict(x_test)
        
    def getAccu(self, x_test, y_test):
        """Return accuracy of fitted model on test data

        Args:
            x_test (list): test data
            y_test (list): test labels

        Returns:
            int : accuracy
        """
        
        
        return accuracy_score(y_test, self.clf.predict(x_test))
    
    def getMcc(self, x_test, y_test):
        """Return accuracy of fitted model on test data

        Args:
            x_test (list): test data
            y_test (list): test labels

        Returns:
            int : accuracy
        """
        
        
        return matthews_corrcoef(y_test, self.clf.predict(x_test))
    
    def getF1(self, x_test, y_test, average=False):
        """Return accuracy of fitted model on test data

        Args:
            x_test (list): test data
            y_test (list): test labels
            average (Boolean): Return mean of g score of each class

        Returns:
            int : accuracy
        """
        if average:
            return geometric_mean_score(y_test, self.clf.predict(x_test),average='macro')
        
        
        return geometric_mean_score(y_test, self.clf.predict(x_test))
    
    
    def getG(self, x_test, y_test, average=False):
        """Return accuracy of fitted model on test data

        Args:
            x_test (list): test data
            y_test (list): test labels
            average (Boolean): Return mean of g score of each class

        Returns:
            int : accuracy
        """
        if average:
            return f1_score(y_test, self.clf.predict(x_test),average='macro')
        
        
        return f1_score(y_test, self.clf.predict(x_test))
    
    def getPerf(self, x_test, y_test, average = False):
        """Return all metrics

        Args:
            x_test (_type_): _
            y_test (_type_): 
            average (bool, optional): Average metrics by class Defaults to False.

        Returns:
            list: all metric (accu, mcc, f1, g)
        """
        y_pred = self.clf.predict(x_test)
        
        if average:
            return accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred),f1_score(y_test, y_pred, average = 'macro'),geometric_mean_score(y_test, y_pred, average='macro')
        
        return accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred),f1_score(y_test, y_pred),geometric_mean_score(y_test, y_pred)
        
        
