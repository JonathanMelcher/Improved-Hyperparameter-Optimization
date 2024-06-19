#This program generates results for the generic function for quasiperiodic tilings and saves them into a file

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

################checking if a point lies on the square

def if_square(v, d, k):
  #v - the vector
  #d - the number of dimensions of v
  #k - length of the square
  for i in range(d):
    if v[i]>k/2 or v[i]<-k/2:
      return False
  return True

################distance between two points

def dist(a, am, m, t):
    #a - first vector
    #am - the m-th element
    #m - the index of this element
    result = 0
    for i in range(m):
        result += 1/np.linalg.norm(am-a[i])
        result += 1/np.linalg.norm(am+a[i])
    for i in range(m+1, t):
        result += 1/np.linalg.norm(am-a[i])
        result += 1/np.linalg.norm(am+a[i])
    return result

################function used by inverting some of the vectors

def spread(x):
  result = 0
  for i in range(len(x)):
    for j in range(i):
      result += 1/np.linalg.norm(x[i]-x[j])
  return result  

################making the vectors

def make_vectors(d, t):
    #d - number of dimensions
    #t - the number of star vectors
    if(d != 2):
        n = 10000*t*d
        T = 10*t/d
        t0 = T/(n-1)
        x = [[random.uniform(-1, 1) for j in range(d)] for k in range(t)]
        x = np.array(x)
        for j in range(t):
                x[j] = x[j]/np.linalg.norm(x[j])

        for j in range(n):
            for i in range(t):
                a = np.array([random.uniform(-1, 1) for j in range(d)])
                a = a/np.linalg.norm(a)
                dE = dist(x, a, i, t) - dist(x, x[i], i, t)
                if dE<0:
                    x[i] = a
                else:
                    if j != n-1:
                        if random.choices([0, 1], weights = (1-np.exp(-dE/T), np.exp(-dE/T)))[0]:
                            x[i] = a
            T = T-t0
        x0 = x
        minimum = spread(x)
        for p in product([1, -1], repeat=t):
          x1 = np.array([x[i]*list(p)[i] for i in range(t)])
          if spread(x1) < minimum:
            x0 = x1
            minimum = spread(x1)

    else:
        if t%2==1:
            x = [[np.cos(2*np.pi*i/t), np.sin(2*np.pi*i/t)] for i in range(t)]
        else:
            x = [[np.cos(np.pi*i/t), np.sin(np.pi*i/t)] for i in range(t)]
        x0 = np.array(x)
    return x0

################making the nD quasiperiodic :

def make_quasi_nD(number, k, d, t, x):
    #k - the dimensions of the square
    #d - number of dimensions
    #t - the number of star vectors
    #number - the number of points
    #m - the number of planes in each direction
    #x - the star vectors
    #z - YES or NO - whether to add points at random      


    list1 = [i for i in range(t)]

    ##assesing the density

    nu = 0 #the numerator
    de = 0 #the denominator 
    
    for c in combinations(list1, d):
      dete = abs(np.linalg.det(np.array([x[i] for i in c])))
      nu += dete
      de += dete**2
      

    den = nu/de #the density


    si = (number/den)**(1/d) 
    factor = si/k 
    m = math.ceil(d*si*np.sqrt(d)/t)

    l = [] #shifts of lines
    [l.append(random.uniform(-1, 1)) for i in range(t-1)]
    l.append(-sum(l))
    l = np.array(l)


    #checking if a point doesn't lie outside the boundary
    #point is a result of an assignment operation

    def check(assigned):
        if assigned in pointssegments:
            return 0
        for i in range(t):
            if assigned[i] > m or assigned[i] <= -m: 
                return 0
        return 1

    def transform(a):
        return np.ndarray.tolist(np.sum([x[i]*a[i] for i in range(t)], axis = 0))

    def assign(a):
        return [math.ceil(sum(a[i]*x[j, i] for i in range(d)) + l[j]) for j in range(t)]

    #sampling all the points


    listquasi = []
    pointssegments = []
    h2 = []

    for c in combinations(list1, d):
        matrix = np.array([x[i] for i in c])
        for ms in product(range(-m, m+1), repeat = d):
            p = np.dot([ms[i]-l[c[i]] for i in range(d)], np.linalg.inv(matrix.transpose()))
            h = assign(p)
            for i in range(d):
                h[c[i]]=ms[i]
            for s in product([0, 1], repeat = d):
                h2 = list(h)
                for i in range(d):
                    h2[c[i]] += s[i]                
                if check(h2)==1:
                    pointssegments.append(h2)
                    listquasi.append(transform(h2))
    
     
    listquasi2 = []
    
    for i in range(len(listquasi)):
      p = listquasi[i]
      if if_square(p, d, si):
        for xy in range(d):
          p[xy] = p[xy]/factor
        listquasi2.append(p)
    y = len(listquasi2)
    
    return listquasi2
################checking the tiling

def checktiling(someth):
    r = f(someth[0], point, A, w)
    for t0 in someth:
        if f(t0, point, A, w) < r:
            r=f(t0, point, A, w)
    return r
    
################making the comparison

k=1
#The number of trials
trials = 10000

print("This will only be the comparison for quasiperiodic patterns")

#Checked number of dimensions
d = int(input("The number of dimensions (3 or 4): "))

#values up to n**d will be checked
if d==2:
    N=20
else:
    N=5

if d==2:
    t=5
if d==3:
    t=5
if d==4:
    t=6
if d==5:
    t=7
print(t, "star vectors")


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

        x = make_vectors(d, t)
        list_pattern = make_quasi_nD(n**d, k, d, t, x)          
        
        for i in range(trials):
          
          A = data_trials[i][0]
          w = data_trials[i][1]
          point = data_trials[i][2]

          scores[0].append(checktiling(list_pattern))
          numbers[0].append(len(list_pattern))
          
          if i%1000==0:
              print("TRIAL ", i, " FINISHED")
                                
       
        Everything_scores.append(scores)
        Everything_numbers.append(numbers)

        licznik += 1

        print("COMPARISON FOR ", n**d, " POINTS FINISHED")



pickle.dump(Everything_scores, open("Quasiperiodic_scores_d_" + str(d) + "_star_" + str(t) + ".txt", 'wb'))
pickle.dump(Everything_numbers, open("Quasiperiodic_numbers_d_" + str(d) + "_star_" + str(t) + ".txt", 'wb'))


print("COMPARISON FINISHED")























 

