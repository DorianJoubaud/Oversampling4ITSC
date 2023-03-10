from iterativeOS import *
import random
import sys
import os


dataset = sys.argv[1]
clfs = sys.argv[2]
tech = sys.argv[3]
step = int(sys.argv[4])

trig = False
nb_iter = 3

x_train = np.array(pd.read_csv(f'data/{dataset}/{dataset}_TRAIN.tsv', delimiter='\t',header=None))
x_test = np.array(pd.read_csv(f'data/{dataset}/{dataset}_TEST.tsv', delimiter='\t',header=None))


out = f'results/{dataset}/{clfs}_{tech}_{step}'
if not os.path.exists(out):
    os.makedirs(out)

np.savetxt(out+'/'+'accI.txt', [0 for i in range(nb_iter)])
np.savetxt(out+'/'+'mccI.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'f1I.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'gI.txt',[0 for i in range(nb_iter)])

np.savetxt(out+'/'+'accB.txt', [0 for i in range(nb_iter)])
np.savetxt(out+'/'+'mccB.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'f1B.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'gB.txt',[0 for i in range(nb_iter)])

y_train = x_train[:,0]
x_train = x_train[:,1:]

y_test = x_test[:,0]
x_test = x_test[:,1:]

clf = Classif(clfs)

y_train = class_offset(y_train, dataset)
y_test = class_offset(y_test, dataset)

_, dist = np.unique(y_train, return_counts = True)

if list(dist).count(dist[0]) == len(dist):
    print('==== Imbalanced initial dataset ====')

for i in range(step):
   
    
    
    
    class_to_keep = 0 # Majority class to keep
    
    # Select all other data
    index_to_keep = np.where(np.array(y_train) == class_to_keep)[0]
    y_reduction = np.delete(y_train, index_to_keep)
    
    
    _, red_dist = np.unique(y_reduction, return_counts = True)
  
    
    tmp_ai = list()
    tmp_mi = list()
    tmp_fi = list()
    tmp_gi = list()
    
    tmp_ab = list()
    tmp_mb = list()
    tmp_fb = list()
    tmp_gb = list()
    
    # We realise 10 time the classification 
    print(f'============== {i/step *100} % ================')
    for ite in range(nb_iter):
        
        correct_dist = False
        # Select % random data belonging to the other classes
        while not correct_dist:
            y_red = random.choices(y_reduction, k = int(len(y_reduction) - i/step*len(y_reduction)))
            _, new_dist = np.unique(y_red, return_counts = True)
            correct_dist = True
            for yi in range(len(new_dist)):
                if new_dist[yi] > red_dist[yi]:
                    correct_dist = False
        
        
        
        
        print(new_dist)
        
        if np.sum(new_dist) <= 2*len(new_dist):
            print('No more :)')
            trig = True
            break
        for idx in range(len(new_dist)):
            
            if (new_dist[idx]< 3):
                new_dist[np.argmax(new_dist)] -= 3 - new_dist[idx]    
                new_dist[idx] = 3
                    
        # Create dict of the new distribution
        sp_str = {0 : list(y_train).count(class_to_keep)} 
        
        bal_str = {idx : list(y_train).count(class_to_keep) for idx in range(len(new_dist)+1)}
        
        for j in range(len(new_dist)):
            sp_str[j+1] = new_dist[j]
        
        print(sp_str)
        x,y = RandomUnderSampler(sampling_strategy=sp_str).fit_resample(x_train,y_train) # x and y
        
        clf.fit(x,y) # Imbalanced classif
        
        a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
        tmp_ai.append(a)
        tmp_mi.append(m)
        tmp_fi.append(f)
        tmp_gi.append(g)
        
        #Balancing phase
        sampler = Sampler(tech, bal_str)
        x_os, y_os = sampler.sampleData(x,y)
        
        
        clf.fit(x_os, y_os) #Balanced classif
        
        a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
        tmp_ab.append(a)
        tmp_mb.append(m)
        tmp_fb.append(f)
        tmp_gb.append(g)
        
    if not trig:
        a = list(np.loadtxt(out+'/'+'accI.txt'))
        m = list(np.loadtxt(out+'/'+'mccI.txt'))
        f = list(np.loadtxt(out+'/'+'f1I.txt'))
        g = list(np.loadtxt(out+'/'+'gI.txt'))
        
    
        a = np.vstack([a, tmp_ai])
        m = np.vstack([m, tmp_mi])
        f = np.vstack([f, tmp_fi])
        g = np.vstack([g, tmp_gi])
        
    
        
        
        
    
        np.savetxt(out+'/'+'accI.txt', a)
        np.savetxt(out+'/'+'mccI.txt' ,m)
        np.savetxt(out+'/'+'f1I.txt', f)
        np.savetxt(out+'/'+'gI.txt', g)
    
        
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