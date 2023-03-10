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
                        
                        
                        selected = x[np.random.choice(np.where(np.array(y) == i)[0])] # Select one random data with the label i
                        
                        selected = tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=2).augment(selected)
                        x.append(selected)
                        y.append(i)
            return x, y
        
        elif(self.sampler == 'SMOTE'):
            return SMOTE(k_neighbors=2, sampling_strategy=self.strg).fit_resample(x,y)
        
        elif(self.sampler == 'ADASYN'):
            return ADASYN(n_neighbors=2, sampling_strategy=self.strg).fit_resample(x,y)
            
            
            
            
            
        

class Classif:
    def __init__(self, clf):
        
        if (clf == 'SVM'):
            
            self.clf = TimeSeriesSVC(kernel="gak", gamma=.1, n_jobs=-1)
            
              
        
    def fit(self, x_train, y_train):
        """
        Fit the model on training data
        
        """
        
        self.clf.fit(x_train, y_train)
        
    
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
        
        return accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred),f1_score(y_test, y_pred, average = None),geometric_mean_score(y_test, y_pred, average = None)
        
        

def class_offset(y, dataset):
    return (y + class_modifier_add(dataset)) * class_modifier_multi(dataset)

def class_modifier_add(dataset):
    if dataset == "FiftyWords":
        return -1 #270
    if dataset == "Adiac":
        return -1 #176
    if dataset == "ArrowHead":
        return 0 #251
    if dataset == "Beef":
        return -1 #470
    if dataset == "BeetleFly":
        return -1 #512
    if dataset == "BirdChicken":
        return -1 #512
    if dataset == "Car":
        return -1 #577
    if dataset == "CBF":
        return -1 #128
    if dataset == "ChlorineConcentration":
        return -1 #166
    if dataset == "CinCECGTorso":
        return -1 #1639
    if dataset == "Coffee":
        return 0 #286
    if dataset == "Computers":
        return -1 #720
    if dataset == "CricketX":
        return -1 #300
    if dataset == "CricketY":
        return -1 #300
    if dataset == "CricketZ":
        return -1 #300
    if dataset == "DiatomSizeReduction":
        return -1 #345
    if dataset == "DistalPhalanxOutlineAgeGroup":
        return -1 #80
    if dataset == "DistalPhalanxOutlineCorrect":
        return 0 #80
    if dataset == "DistalPhalanxTW":
        return -3 #80
    if dataset == "Earthquakes":
        return 0 #512
    if dataset == "ECG200":
        return 1 #96
    if dataset == "ECG5000":
        return -1 #140
    if dataset == "ECGFiveDays":
        return -1 #136
    if dataset == "ElectricDevices":
        return -1 #96
    if dataset == "FaceAll":
        return -1 # 131
    if dataset == "FaceFour":
        return -1 # 350
    if dataset == "FacesUCR":
        return -1 # 131
    if dataset == "Fish":
        return -1 # 463
    if dataset == "FordA":
        return 1 #500
    if dataset == "FordB":
        return 1 # 500
    if dataset == "GunPoint":
        return -1 # 150
    if dataset == "Ham":
        return -1 # 431
    if dataset == "HandOutlines":
        return 0 # 2709
    if dataset == "Haptics":
        return -1 # 1092
    if dataset == "Herring":
        return -1 # 512
    if dataset == "InlineSkate":
        return -1 # 1882
    if dataset == "InsectWingbeatSound":
        return -1 # 256
    if dataset == "ItalyPowerDemand":
        return -1 # 24
    if dataset == "LargeKitchenAppliances":
        return -1 # 720
    if dataset == "Lightning2":
        return 1 # 637
    if dataset == "Lightning7":
        return 0 # 319
    if dataset == "Mallat":
        return -1 # 1024
    if dataset == "Meat":
        return -1 # 448
    if dataset == "MedicalImages":
        return -1 # 99
    if dataset == "MiddlePhalanxOutlineAgeGroup":
        return -1 #80
    if dataset == "MiddlePhalanxOutlineCorrect":
        return 0 #80
    if dataset == "MiddlePhalanxTW":
        return -3 #80
    if dataset == "MoteStrain":
        return -1 #84
    if dataset == "NonInvasiveFetalECGThorax1":
        return -1 #750
    if dataset == "NonInvasiveFetalECGThorax2":
        return -1 #750
    if dataset == "OliveOil":
        return -1 #570
    if dataset == "OSULeaf":
        return -1 #427
    if dataset == "PhalangesOutlinesCorrect":
        return 0 #80
    if dataset == "Phoneme":
        return -1 #1024
    if dataset == "Plane":
        return -1 #144
    if dataset == "ProximalPhalanxOutlineAgeGroup":
        return -1 #80
    if dataset == "ProximalPhalanxOutlineCorrect":
        return 0 #80
    if dataset == "ProximalPhalanxTW":
        return -3 #80
    if dataset == "RefrigerationDevices":
        return -1 #720
    if dataset == "ScreenType":
        return -1 #720
    if dataset == "ShapeletSim":
        return 0 #500
    if dataset == "ShapesAll":
        return -1 # 512
    if dataset == "SmallKitchenAppliances":
        return -1 #720
    if dataset == "SonyAIBORobotSurface2":
        return -1 #65
    if dataset == "SonyAIBORobotSurface1":
        return -1 #70
    if dataset == "StarLightCurves":
        return -1 #1024
    if dataset == "Strawberry":
        return -1 #235
    if dataset == "SwedishLeaf":
        return -1 # 128
    if dataset == "Symbols":
        return -1 #398
    if dataset == "SyntheticControl":
        return -1 #60
    if dataset == "ToeSegmentation1":
        return 0 #277
    if dataset == "ToeSegmentation2":
        return 0 #343
    if dataset == "Trace":
        return -1 #275
    if dataset == "TwoLeadECG":
        return -1 #82
    if dataset == "TwoPatterns":
        return -1 #128
    if dataset == "UWaveGestureLibraryX":
        return -1 # 315
    if dataset == "UWaveGestureLibraryY":
        return -1 # 315
    if dataset == "UWaveGestureLibraryZ":
        return -1 # 315
    if dataset == "UWaveGestureLibraryAll":
        return -1 # 945
    if dataset == "Wafer":
        return 1 #152
    if dataset == "Wine":
        return -1 #234
    if dataset == "WordSynonyms":
        return -1 #270
    if dataset == "Worms":
        return -1 #900
    if dataset == "WormsTwoClass":
        return -1 #900
    if dataset == "Yoga":
        return -1 #426

    if dataset == "ACSF1":
        return 0
    if dataset == "AllGestureWiimoteX":
        return -1
    if dataset == "AllGestureWiimoteY":
        return -1
    if dataset == "AllGestureWiimoteZ":
        return -1
    if dataset == "BME":
        return -1
    if dataset == "Chinatown":
        return -1
    if dataset == "Crop":
        return -1
    if dataset == "DodgerLoopDay":
        return -1
    if dataset == "DodgerLoopGame":
        return -1
    if dataset == "DodgerLoopWeekend":
        return -1
    if dataset == "EOGHorizontalSignal":
        return -1
    if dataset == "EOGVerticalSignal":
        return -1
    if dataset == "EthanolLevel":
        return -1
    if dataset == "FreezerRegularTrain":
        return -1
    if dataset == "FreezerSmallTrain":
        return -1
    if dataset == "Fungi":
        return -1
    if dataset == "GestureMidAirD1":
        return -1
    if dataset == "GestureMidAirD2":
        return -1
    if dataset == "GestureMidAirD3":
        return -1
    if dataset == "GesturePebbleZ1":
        return -1
    if dataset == "GesturePebbleZ2":
        return -1
    if dataset == "GunPointAgeSpan":
        return -1
    if dataset == "GunPointMaleVersusFemale":
        return -1
    if dataset == "GunPointOldVersusYoung":
        return -1
    if dataset == "HouseTwenty":
        return -1
    if dataset == "InsectEPGRegularTrain":
        return -1
    if dataset == "InsectEPGSmallTrain":
        return -1
    if dataset == "MelbournePedestrian":
        return -1
    if dataset == "MixedShapesRegularTrain":
        return -1
    if dataset == "MixedShapesSmallTrain":
        return -1
    if dataset == "PickupGestureWiimoteZ":
        return -1
    if dataset == "PigAirwayPressure":
        return -1
    if dataset == "PigArtPressure":
        return -1
    if dataset == "PigCVP":
        return -1
    if dataset == "PLAID":
        return 0
    if dataset == "PowerCons":
        return -1
    if dataset == "Rock":
        return -1
    if dataset == "SemgHandGenderCh2":
        return -1
    if dataset == "SemgHandMovementCh2":
        return -1
    if dataset == "SemgHandSubjectCh2":
        return -1
    if dataset == "ShakeGestureWiimoteZ":
        return -1
    if dataset == "SmoothSubspace":
        return -1
    if dataset == "UMD":
        return -1
    return 0

def class_modifier_multi(dataset):
    if dataset == "ECG200":
        return 0.5 #96
    if dataset == "FordA":
        return 0.5 #500
    if dataset == "FordB":
        return 0.5 # 500
    if dataset == "Lightning2":
        return 0.5 # 637
    if dataset == "Wafer":
        return 0.5 #152
    return 1