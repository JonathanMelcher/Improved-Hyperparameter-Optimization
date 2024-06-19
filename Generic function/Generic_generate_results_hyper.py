#This program generates results for the generic function for hyperuniform patterns (in 3-5 dimensions) and saves them into a file

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations, product
from mpl_toolkits import mplot3d
import pickle

random.seed(100)
np.random.seed(100)

################our function

def f(x, point, A, w):
    #x - the vector
    #A - the matrix making a linear transformation of the coordinates
    #w - the vector of weights
    #p - the point from which we calculate the distance

    return np.linalg.norm(np.dot([x[i] - point[i] for i in range(d)], A)*w)**2

################transforming a tiling to required maximum/minimum

def transform_tiling(tiling, old_min, old_max, new_min, new_max, d):
    #d is the number of dimensions
    #old_min is the vector with the old minimum, old_max - maximum
    #new_min sis the new minimum, new_max - maximum
    for i in range(d):
        a = (new_max[i]-new_min[i])/(old_max[i]-old_min[i])
        b = (new_min[i]*old_max[i]-old_min[i]*new_max[i])/(old_max[i]-old_min[i])
        for point in tiling:
            point[i] = a*point[i]+b

################accessing the file and making a hyperuniform pattern

def get_hyper(d, directory):
    #k - the sidelength of the square
    #number - the number if points
    #d - the number of dimensions


    x = []
    with open(directory) as f:
        line = f.readline()
        while line:
            x.append([float(i) for i in line.split()])
            x[-1] = x[-1][:d]
            line = f.readline()

    density = 0 #the density

    for p in x:
        if all([p[i]>=0.05 and p[i]<=0.95 for i in range(d)]):
            density += 1


    density = density/(0.9**d)

    return x, density

def make_hyper(number, k, d, pattern = [], density = 0, s=0):

    if d>2:

        side = (number/density)**(1/d)
        
        if 0.95-side/2 >= 0.5:
            center = [random.uniform(0.95-side/2, 0.05+side/2) for i in range(d)]
        else:
            center = [0.5 for i in range(d)]


        listhyper = []
        for p in pattern:
            if all([p[i]-center[i] < side/2 and p[i]-center[i] > -side/2 for i in range(d)]):
                   listhyper.append(list(p))


        new_min = [-k/2 for i in range(d)]
        new_max = [k/2 for i in range(d)]
        old_min = [center[i]-side/2 for i in range(d)]
        old_max = [center[i]+side/2 for i in range(d)]
                   
        transform_tiling(listhyper, old_min, old_max, new_min, new_max, d)

        return listhyper
    
    else:
        #s  - stealth

        if s==0.49 or s==0.38:
            run = int(random.uniform(0, 60))
        if s==0.13 or s==0.25:
            run = int(random.uniform(0, 20))     

        x = []
        if s==0.38 and run>=20:
         o = r"..\stealthy-point-patterns\stealthy-"+str(s)+r"-1000\stealthy-"+str(s)+r"-lbfgs-1000-run-"+str(run)+".dat"
        else:
         o = r"..\stealthy-point-patterns\stealthy-"+str(s)+r"-1000\stealthy-"+str(s)+r"-lbfgs-1000-run-"+str(run)+".txt"
        with open(o) as f:
            line = f.readline()
            while line:
                x.append([float(i) for i in line.split()])
                line = f.readline()

        d = 0 #the density
        listhyper = []

        for p in x:
            if p[0] <= 20 and p[1] <= 20 and p[0] >= 5 and p[1] >= 5:
                d += 1
        d = d/225
        a = np.sqrt(number/d)
        f1 = k/a
        f2 = (a+4)*k/(2*a)
        for p in x:
            if p[0] <= a+2 and p[1] <= a+2 and p[1] >= 2 and p[0] >= 2:
                listhyper.append([p[0]*f1 - f2, p[1]*f1 - f2])

        return listhyper

################checking the tiling

def checktiling(someth):
    r = f(someth[0], point, A, w)
    for t0 in someth:
        if f(t0, point, A, w) < r:
            r=f(t0, point, A, w)
    return r
        
################making the comparison


s=0.49
k=1

#The number of trials
trials = 10000

print("This will only be the comparison for hyperuniform patterns")

#Checked number of dimensions
d = int(input("The number of dimensions (2, 3, 4, or 5): "))

#values up to n**d will be checked
if d==2:
    N=20
else:
    N=5
stealth = float(input("The stealth - 0.4 or 0.3 for >2D; 0.49, 0.38, 0.25,or 0.13 for 2D: "))


numbers = [[]] 

results = [[]]

rev = [(N-i) for i in range(N-1)]

Everything_scores = []
Everything_numbers = []

with open("Data_trials_d_" + str(d) + ".p", 'rb') as czytaj:
    data_trials_all = pickle.load(czytaj)

if d==3:
    directory0 = r"..\stealthy-point-patterns2\3\5000\0.3\0\positions.dat"
if d==4:
    directory0 = r"..\stealthy-point-patterns2\4\5000\0.3\0\positions.dat"
if d==5:
    directory0 = r"..\stealthy-point-patterns2\5\5000\0.3\0\positions.dat"

if d!=2:
    hyperall, densityhyper = get_hyper(d, directory0)


licznik=0


for n in rev:

        data_trials = data_trials_all[licznik]
        
        scores = [[]]
        numbers = [[]]
        for i in range(trials):

          listhyper = []
          
          while len(listhyper)==0:
              if d==2:
                  listhyper = make_hyper(n**d, k, d, s=stealth)
              else:
                  listhyper = make_hyper(n**d, k, d, hyperall, densityhyper)


          A = data_trials[i][0]
          w = data_trials[i][1]
          point = data_trials[i][2]

          scores[0].append(checktiling(listhyper))
          numbers[0].append(len(listhyper))
          
          if i%1000==0:
              print("TRIAL ", i, " FINISHED")
                                
       
        Everything_scores.append(scores)
        Everything_numbers.append(numbers)

        licznik += 1

        print("COMPARISON FOR ", n**d, " POINTS FINISHED")



pickle.dump(Everything_scores, open("Hyperuniform_scores_" + str(stealth) + "_d_" + str(d) + ".txt", 'wb'))
pickle.dump(Everything_numbers, open("Hyperuniform_numbers_" + str(stealth) + "_d_" + str(d) + ".txt", 'wb'))


print("COMPARISON FINISHED")























 

