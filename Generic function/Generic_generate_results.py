#This program generates results for the generic function and saves them into a file

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations, product
from mpl_toolkits import mplot3d
import pickle

random.seed(100)
np.random.seed(100)

################generic function used in the comparison

def f(x, point, A, w):
    #x - the vector
    #A - the matrix making a linear transformation of the coordinates
    #w - the vector of weights
    #p - the point from which we calculate the distance

    return np.linalg.norm(np.dot([x[i] - point[i] for i in range(d)], A)*w)**2

################transforming a tiling to required maximum/minimum of hyperparameter space

def transform_tiling(tiling, old_min, old_max, new_min, new_max, d):
    #d is the number of dimensions
    #old_min is the vector with the old minimum, old_max - maximum
    #new_min is the new minimum, new_max - maximum
    for i in range(d):
        a = (new_max[i]-new_min[i])/(old_max[i]-old_min[i])
        b = (new_min[i]*old_max[i]-old_min[i]*new_max[i])/(old_max[i]-old_min[i])
        for point in tiling:
            point[i] = a*point[i]+b

################making a hyperuniform point pattern

def make_hyper(k, number, s, run):
    #number - the intended number of points
    #k - the side length of the square
    #s  - stealth of the hyperuniform pattern
    #run - number referring to the name of a file with the hyperuniform pattern

    x = []
    if s==0.38 and run>=20:
     o = r"../stealthy-point-patterns\stealthy-"+str(s)+r"-1000\stealthy-"+str(s)+r"-lbfgs-1000-run-"+str(run)+".dat"
    else:
     o = r"../stealthy-point-patterns\stealthy-"+str(s)+r"-1000\stealthy-"+str(s)+r"-lbfgs-1000-run-"+str(run)+".txt"
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


################making a grid with a given number of points

def make_grid(number, k, d):
 #number - the intended number of points   
 #k - the length of the square
 #d - the number of dimensions
  gr = pow(number, 1/d)
  if math.ceil(gr)**d == number:
    gr = math.ceil(gr)
  else:
    gr = int(gr)
  g = np.linspace(-k/2, k/2, gr)
  listgrid = []
  for c in product(g, repeat = d):
       listgrid.append(list(c))
  return listgrid

################making a random pattern with a given number of points

def make_random(number, k, d):
    #number - the intended number of points
    #k - the side length of the squares
    #d - the number of dimensions
    listrand = []
    for r in range(number):
      listrand.append([random.uniform(-k/2, k/2) for j in range(d)])
    return listrand

################making a gridrandom pattern with a given number of points

def make_gridrandom(number, k, d):
    #number - the intended number of points
    #k - the length of the squares
    #d - the number of dimensions
    listgridrand = []
    gr = pow(number, 1/d)
    if math.ceil(gr)**d == number:
      gr = math.ceil(gr)
    else:
      gr = int(gr)
    g = np.linspace(-k/2, k/2, gr+1)
    for c in product(g[:gr], repeat=d):
          listgridrand.append([c[j]+random.uniform(0, k/(gr)) for j in range(d)])
    return listgridrand

################checking if a point lies on the square

def if_square(v, d, k):
  #v - the vector
  #d - the number of dimensions of v
  #k - length of the square
  for i in range(d):
    if v[i]>k/2 or v[i]<-k/2:
      return False
  return True

################"distance" function, used in generating star vectors for the quasiperiodic tiling

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

################another function used for generating star vectors for the quasiperiodic tiling

def spread(x):
  result = 0
  for i in range(len(x)):
    for j in range(i):
      result += 1/np.linalg.norm(x[i]-x[j])
  return result  

################generating spread out star vectors for the quasiperiodic tilings

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
    print("Star in", d, "dimensions with", t, "star vectors generated")
    return x0

################making the quasiperiodic tiling with a given number of points

def make_quasi_nD(number, k, d, t, x):
    #number - the intended number of points
    #k - the side length of the square
    #d - number of dimensions
    #t - the number of star vectors
    #x - the star vectors

    list1 = [i for i in range(t)]

    ##assesing the density of the tiling

    nu = 0 #the numerator
    de = 0 #the denominator 
    
    for c in combinations(list1, d):
      dete = abs(np.linalg.det(np.array([x[i] for i in c])))
      nu += dete
      de += dete**2
      

    den = nu/de #the density

    ##calculating m

    si = (number/den)**(1/d) #the initial side length of the square
    factor = si/k #the scaling factor
    m = math.ceil(d*si*np.sqrt(d)/t)

    ##the rest

    l = [] #shifts of lines
    [l.append(random.uniform(-1, 1)) for i in range(t-1)]
    l.append(-sum(l))
    l = np.array(l)


    #checking if a point (already "assigned") lies in the permitted boundary

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

################making the RSA pattern

def make_RSA(number, Length, d):
    #number - expected number of points 
    #Length - side length of the square
    #d - number of dimensions

    ##saturation density
    def density_RSA(d):
      if d==1:
        return 0.7475
      if d==2:
        return 0.5470735
      if d==3:
        return 0.3841307
      if d==4:
        return 0.2600781
      if d==5:
        return 0.1707761
      if d==6:
        return 0.109302
      if d==7:
        return 0.068404
      if d==8:
        return 0.0423
      return 1/2**(d)

    ##volume of a d-dimensional unit ball
    def ball(d):
      if d==0:
        return 1
      if d==1:
        return 2
      return 2*np.pi*ball(d-2)/d

    D0 = 2*Length*(density_RSA(d)/(number*ball(d)))**(1/d) #diameter of the spheres
    if d==2 or d==3:
      D = D0/(1-D0/(1.5*Length))
    if d==4:
      D = D0/(1-D0/(1.65*Length))
    if d>=5 and d<=7:
      D = D0/(1-D0/(2*Length))
    if d==8:
      D = D0/(1-D0/(Length))

    if D>Length:
      D=D0
    
    bigvoxel_amount = int(Length//D)
    bigvoxel = Length/(Length//D) #bigvoxel size
    v = Length/(Length//(D)) #smallvoxel initial sizes
    N = int(Length//(D)) #initial amount of smallvoxels

    ##for the neighborhood list

    n_shape = [bigvoxel_amount for i in range(d)]
    n = np.empty(n_shape, dtype = object)
    n.fill([])
  
    ##checking whether a sphere with center c can be added

    def check_sphere(c):
      c1 = c/bigvoxel
      c1 = c1.astype(int)
      for t in product([0, -1, 1], repeat = d):
        try:
            for a in n[tuple(c1 + np.array(t))]:
              if np.linalg.norm(a - c) < D:
                return False
        except IndexError:
            continue
      n[tuple(c1)].append(c)
      return True

    ##checking whether a voxel with corner c and length v can be discarded

    def check_voxel(c, vec0, vec0_dist):
      c1 = c/bigvoxel
      c1 = c1.astype(int)
      for t in product([0, -1, 1], repeat = d):
        try:
            for a0 in n[tuple(c1 + np.array(t))]:
              if np.linalg.norm(a0-vec0-c) < D-vec0_dist:
                return False
        except IndexError:
            continue
      return True

    count = 0

    list_RSA = []

    #maximum number od trials after which we proceed to the next step
    count_MAX = 30

    #vector from corner to center of a voxel
    vec = np.array([v/2 for i in range(d)])
    vec_dist = np.linalg.norm(vec)

    vlist = []

    ##phase 1 - searching through all space
    while count < count_MAX:
      c = np.array([random.uniform(0, Length) for i in range(d)])
      if check_sphere(c):
        list_RSA.append(c)
        count=0
      else:
        count+=1

    print("Phase 1 complete")

    ##setup for the second phase - creating the voxel list
    l = [i*v for i in range(N)]
    for t in product(l, repeat = d):
      if check_voxel(np.array(t), vec, vec_dist):
        vlist.append(np.array(t))
    vnum = len(vlist)

    ##phase 2 - iterating until the voxel list is empty 

    while len(vlist) >= 1:
      count=0
      while count < count_MAX:
        i = int(random.uniform(0, vnum))
        c = np.array([random.uniform(0, v) for j in range(d)]) + vlist[i]
        if check_sphere(c):
          count=0
          list_RSA.append(c)
        else:
          count+=1
      print("Current value of 2^(d) * (length of smallvoxel list)", 2**d * len(vlist))

      if d>=4 and d<=4 and 2**d * len(vlist) > 100000:
        vlist1 = []
        for voxel in vlist:
          if check_voxel(voxel, vec, vec_dist):
            vlist1.append(voxel)
        vlist = list(vlist1)
              
        print("Moving to phase 3")
        break

      if 5<=d and 2**d * len(vlist) > 10000:
        vlist1 = []
        for voxel in vlist:
          if check_voxel(voxel, vec, vec_dist):
            vlist1.append(voxel)
        vlist = list(vlist1)
              
        print("Moving to phase 3")
        break
      
      v, vec, vec_dist = v/2, vec/2, vec_dist/2

      vlist1 = []

      for voxel in vlist:
        if check_voxel(voxel, 2*vec, 2*vec_dist):
          for t in product([0, v], repeat=d):
            if check_voxel(voxel+np.array(t), vec, vec_dist):
              vlist1.append(voxel+np.array(t))
      vlist=list(vlist1)
      vnum=len(vlist)

    if len(vlist)>=1:
      count_max = max(int(500000/len(vlist)), 30)
    
    random.shuffle(vlist)

    ##phase 3

    while len(vlist)>=1:
        count = 0
        while count<=count_max:
          c = np.array([random.uniform(0, v) for j in range(d)]) + vlist[-1]
          if check_sphere(c):
            count=0
            list_RSA.append(c)
          else:
            count+=1
        vlist.pop()
    
    return list_RSA


################generating the Latin Hypercube Sampling pattern with a given number of points

def make_latin(number, k, d):
    #number - intended number of points
    #k - the side length of the square
    #d - the number of dimensions
    
    l = [i for i in range(number)]
    rng = np.random.default_rng()
    m = [1 for i in range(d)]
    m[d-1]=l
    for i in range(d-1):
        m[i] = rng.permutation(l)
    listlatin = []
    for i in range(number):
        listlatin.append([k*(m[j][l[i]])/(number-1)-(k/2) for j in range(d)])
    return listlatin

################checking a point pattern at minimizing a certain function

def checktiling(someth):
    r = f(someth[0], point, A, w)
    for t0 in someth:
      if f(t0, point, A, w) < r:
        r=f(t0, point, A, w)
    return r
    
################making the comparison

#used stealth for the hyperuniform patterns
s=0.49

k=1

#The number of trials
trials = 10000

#Checked number of dimensions
d = int(input("The number of dimensions (2, 3, 4, or 5): "))

#values up to n**d will be checked
if d==2:
    N=20
else:
    N=5

#maximal number of checked star vectors
if d==2:
    mak=8
else:
    mak=3

if d==2:
    incl=1
else:
    incl=0

l = [0 for i in range(d-1)]
a = [0 for i in range(mak+1)]

upper_limit = mak+3
if incl:
    upper_limit = mak+4

scores = [[0] for i in range(upper_limit)]

numbers = [[] for i in range(upper_limit)] #overall numbers of points for each tiling

results = [[] for i in range(upper_limit)] #overall results for a given number of points


listquasiall = [[] for i in range(mak+1)]
lengths = [0 for i in range(mak+1)]


rev = [(N-i) for i in range(N-1)]

y_err = [[] for i in range(upper_limit)]

x_err = [[] for i in range(upper_limit)]

wins = [0 for i in range(upper_limit)]

new_min = np.array([-k/2 for i in range(d)])
new_max = np.array([k/2 for i in range(d)])

old_min = np.array([0 for i in range(d)])
old_max = np.array([k for i in range(d)])

Everything_scores = []
Everything_numbers = []

data_trials = []

for n in rev:
        scores = [[] for i in range(upper_limit)]
        data_trials_temp = []
        
        listgrid = make_grid(n**d, k, d)
        listrand = make_random(n**d, k, d)
        listgridrand = make_gridrandom(n**d, k, d)
        
        for i in range(4, mak+1):
            v = make_vectors(d, i)
            listquasiall[i] = make_quasi_nD(n**d, k, d, i, v)
            lengths[i] = len(listquasiall[i])
        listRSA = make_RSA(n**d, k, d)
        transform_tiling(listRSA, old_min, old_max, new_min, new_max, d)
        listlatin = make_latin(n**d, k, d)
        if incl:
            run = int(random.uniform(0, 60))
            listhyper = make_hyper(k, n**d, s, run)
        for i in range(trials):
          listrand = make_random(len(listRSA), k, d)
          listgridrand = make_gridrandom(n**d, k, d)
          listlatin = make_latin(n**d, k, d)
          if incl:
            run = int(random.uniform(0, 60))
            listhyper = make_hyper(k, n**d, s, run)
          A = [[np.random.uniform(-1, 1) for i1 in range(d)] for i2 in range(d)]
          A = A/np.linalg.norm(A)
          w = np.array([np.random.uniform(0, 1000) for i in range(d)])
          w = 100*w/np.linalg.norm(w)
          
          point = [np.random.uniform(-k/2, k/2) for i in range(d)]

          data_trials_temp.append([A, w, point])
          
          scores[1].append(checktiling(listgrid))
          scores[2].append(checktiling(listrand))
       
          if listgridrand:
            scores[3].append(checktiling(listgridrand))
          for t in range(4, mak+1):
            if listquasiall[t]:
              scores[t].append(checktiling(listquasiall[t]))
             
          scores[mak+1].append(checktiling(listRSA))
          scores[mak+2].append(checktiling(listlatin))
          
          if incl:
              scores[mak+3].append(checktiling(listhyper))

          listhelp = []
          for i in range(1, len(scores)):
              try:
                  listhelp.append(scores[i][-1])
              except IndexError:
                  listhelp.append(1000000000)
          wins[listhelp.index(min(listhelp))+1] += 1
          
        for i in range(1, upper_limit):
          if len(scores[i])>=1:
            results[i].append(sum(scores[i])/len(scores[i]))
            
        numbers[1].append(len(listgrid))
        y_err[1].append(np.std(scores[1])/np.sqrt(len(scores[1])))
        numbers[2].append(len(listrand))
        y_err[2].append(np.std(scores[2])/np.sqrt(len(scores[2])))


        if listgridrand:
          y_err[3].append(np.std(scores[3])/np.sqrt(len(scores[3])))
          numbers[3].append(len(listgridrand))

        for i in range(4, mak+1):
          if len(scores[i])>=1:
            y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))
            numbers[i].append(len(listquasiall[i]))

        i = mak+1  
        y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))
        numbers[i].append(len(listRSA))


        i = mak+2
        y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))
        numbers[i].append(len(listlatin))
        
        if incl:
            i = mak+3
            y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))
            numbers[i].append(len(listhyper))
       
        Everything_scores.append(scores)
        Everything_numbers.append(numbers)
        data_trials.append(data_trials_temp)

        print("Comparison for", len(listgrid), "points finished")



pickle.dump(data_trials, open("Data_trials_d_" + str(d) + ".p", 'wb'))

pickle.dump(Everything_scores, open("Scores_d_" + str(d) + ".txt", 'wb'))
pickle.dump(Everything_numbers, open("Numbers_d_" + str(d) + ".txt", 'wb'))

print("Comparison finished")






















 
