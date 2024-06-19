#This program generates results for the generic function for random search and saves them into a file

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations, product
from mpl_toolkits import mplot3d
import pickle

random.seed(100)
np.random.seed(100)

################the function

def f(x, point, A, w):
    #x - the vector
    #A - the matrix making a linear transformation of the coordinates
    #w - the vector of weights
    #p - the point from which we calculate the distance

    return np.linalg.norm(np.dot([x[i] - point[i] for i in range(d)], A)*w)**2

def g(x, point):
    return np.exp(sum((x[i] - point[i])**2 for i in range(d))**(1/2))

################transforming a pattern to required maximum/minimum

def transform_tiling(tiling, old_min, old_max, new_min, new_max, d):
    #d is the number of dimensions
    #old_min is the vector with the old minimum, old_max - maximum
    #new_min sis the new minimum, new_max - maximum
    for i in range(d):
        a = (new_max[i]-new_min[i])/(old_max[i]-old_min[i])
        b = (new_min[i]*old_max[i]-old_min[i]*new_max[i])/(old_max[i]-old_min[i])
        for point in tiling:
            point[i] = a*point[i]+b

################making a random tiling with a given number of points

def make_random(number, k, d, z="NO"):
    #k - the length of the squares
    #d - the number of dimensions
    #z - do you want to add random points
    listrand = []
    for r in range(number):
      listrand.append([random.uniform(-k/2, k/2) for j in range(d)])
    return listrand

################checking the tiling

def checktiling(someth):
    r = f(someth[0], point, A, w)
    for t0 in someth:
        if f(t0, point, A, w) < r:
            r=f(t0, point, A, w)
    return r

################making the comparison

#index 1 grid
#index 2 random
#index 3 gridrandom


s=0.49
k=1

#The number of trials
trials = 10000

print("This will only be the comparison for random patterns")

#Checked number of dimensions
d = int(input("The number of dimensions (2, 3, 4, or 5): "))

#values up to n**d will be checked
if d==2:
    N=20
else:
    N=5


numbers = [[]] 
results = [[]] 

rev = [(N-i) for i in range(N-1)]

Everything_scores = []
Everything_numbers = []

with open("Data_trials_d_" + str(d) + ".p", 'rb') as czytaj:
    data_trials_all = pickle.load(czytaj)


licznik=0


for n in rev:

        data_trials = data_trials_all[licznik]
        
        scores = [[]]
        numbers = [[]]
        for i in range(trials):


          listrand = make_random(n**d, k, d)
          
          A = data_trials[i][0]
          w = data_trials[i][1]
          point = data_trials[i][2]

          scores[0].append(checktiling(listrand))
          numbers[0].append(len(listrand))
          
          if i%1000==0:
              print("TRIAL ", i, " FINISHED")
                                
       
        Everything_scores.append(scores)
        Everything_numbers.append(numbers)

        licznik += 1

        print("COMPARISON FOR ", n**d, " POINTS FINISHED")


pickle.dump(Everything_scores, open("Random_scores_d_" + str(d) + ".txt", 'wb'))
pickle.dump(Everything_numbers, open("Random_numbers_d_" + str(d) + ".txt", 'wb'))


print("COMPARISON FINISHED")























 

