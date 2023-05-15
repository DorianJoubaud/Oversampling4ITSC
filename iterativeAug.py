from iterativeOS import *
import random
import sys
import os
import ast
import tensorflow as tf
from imbalanced_degree import *

# ======= Input settings =======
dataset = sys.argv[1]
clfs = sys.argv[2]
tech = sys.argv[3]
pourcentage = int(sys.argv[4])


trig = False
nb_iter = 5


# ======= Input data =======
x_train = np.array(pd.read_csv(f'data/{dataset}/{dataset}_TRAIN.tsv', delimiter='\t',header=None).reset_index(drop=True))
x_test = np.array(pd.read_csv(f'data/{dataset}/{dataset}_TEST.tsv', delimiter='\t',header=None).reset_index(drop=True))

# ======= Creating folders for results ======= 
out = f'iterativeBalancing/results/{dataset}/{clfs}_{tech}'
if not os.path.exists(out):
    os.makedirs(out)
    
np.savetxt(out+'/'+'accB.txt', [0 for i in range(nb_iter)])
np.savetxt(out+'/'+'mccB.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'f1B.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'gB.txt',[0 for i in range(nb_iter)])



# ======= Preprocessing data =======
y_train = x_train[:,0]
x_train = x_train[:,1:]

y_test = x_test[:,0]
x_test = x_test[:,1:]

y_train = class_offset(y_train, dataset)
y_test = class_offset(y_test, dataset)

# If pourcentage < the total number to data, we set it to the total number of data - 1
if (pourcentage<=len(y_train)):
    pourcentage = len(y_train)-1
    
nb_timesteps = int(x_train.shape[1] / 1)
input_shape = (nb_timesteps , 1)
    
x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

_, dist = np.unique(y_train, return_counts = True)


bal_str = {i:np.max(dist) for i in range(len(dist))}

y_train = tf.keras.utils.to_categorical(y_train, num_classes=None, dtype="float32")
y_test = tf.keras.utils.to_categorical(y_test, num_classes=None, dtype="float32")

# ======= Create all distribution from balance to imbalance =======

 


all_dist = np.array(balance(dist, pourcentage))
dist_id = list()   

for dist_i in all_dist:
    dist_id.append([dist_i, ID(dist_i, 'HE')])
            




dist_id = np.array(dist_id)

id_trunk = dist_id[:,1]
dist_trunk = dist_id[:,0]



pd.DataFrame(list(zip(dist_trunk, id_trunk))).to_csv(out+'/'+'id.txt')


# ======= Classifier initialization =======
clf = Classif(clfs)

# ======= Warning if initial data is not balanced =======
if list(dist).count(dist[0]) == len(dist):
    print('==== Imbalanced initial dataset ====')
    

    

# ======= Let s go =======





for i in range(len(dist_trunk)):
    print(f'======= {i/len(dist_trunk)*100} % =======')
    # ======= Create the sampling strategy =======
    sp_str = {j:dist_trunk[i][j] for j in range(len(dist))}
    print(sp_str)
    # ======= Temporary list to store the results
    tmp_ai = list()
    tmp_mi = list()
    tmp_fi = list()
    tmp_gi = list()
    
    tmp_ab = list()
    tmp_mb = list()
    tmp_fb = list()
    tmp_gb = list()
    
    # ======= Loop for how many time do we make the classification for one case =======
    for ite in range(nb_iter):
        
        
        # ======= Balancing phase using Data augmentation techniques choosen =======
        sampler = Sampler(tech, bal_str)
        
        x_os, y_os = sampler.sampleData(x_train[:,:,0],np.argmax(y_train, axis = 1))
        
        x_os = np.array(x_os).reshape((-1, input_shape[0], input_shape[1]))
        y_os = tf.keras.utils.to_categorical(y_os, num_classes=None, dtype="float32")
        
        # ======= Classif on balanced data with synthetic data =======
        if not os.path.exists(out+'/B_history'+f'/B_{i}'):
            os.makedirs(out+'/B_history'+f'/B_{i}')
        
        clf.fit(x_os, y_os, x_test, y_test, dataset, f'B_{i}', out=out+'B_history',iters = ite) 
        
        
        a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
        tmp_ab.append(a)
        tmp_mb.append(m)
        tmp_fb.append(f)
        tmp_gb.append(g)
        
        
    # ======= Saving the nb_iter time we made the exp =======
    # Each line of the output file will represent a classification made with different imbalance
    # Each column of the output file will represent the i th of the nb_iter classif we made for a given imbalance
    
    # ======= Balanced data with synthetic results =======
        
    a = list(np.loadtxt(out+'/'+'accB.txt'))
    m = list(np.loadtxt(out+'/'+'mccB.txt'))
    f = list(np.loadtxt(out+'/'+'f1B.txt'))
    g = list(np.loadtxt(out+'/'+'gB.txt'))
        
    a = np.vstack([a, tmp_ab])
    m = np.vstack([m, tmp_mb])
    f = np.vstack([f, tmp_fb])
    g = np.vstack([g, tmp_gb])
        
    np.savetxt(out+'/'+'accB.txt', a)
    np.savetxt(out+'/'+'mccB.txt',m)
    np.savetxt(out+'/'+'f1B.txt',f)
    np.savetxt(out+'/'+'gB.txt',g)
        
       
    

                 
        
    
print('========= END =========')