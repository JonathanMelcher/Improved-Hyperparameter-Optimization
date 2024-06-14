import numpy as np
import random
import math
from itertools import combinations, product




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
    for i in range(m+1, t):
        result += 1/np.linalg.norm(am-a[i])
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

    else:
        if t%2==1:
            x = [[np.cos(2*np.pi*i/t), np.sin(2*np.pi*i/t)] for i in range(t)]
        else:
            x = [[np.cos(np.pi*i/t), np.sin(np.pi*i/t)] for i in range(t)]
        x = np.array(x)
    return x

################making the nD quasiperiodic :

def make_quasi_nD(number, k, d, t, x, z="NO"):
    #k - the dimensions of the square
    #d - number of dimensions
    #t - the number of star vectors
    #number - the number of points
    #m - the number of planes in each direction
    #x - the star vectors
    #z - YES or NO - whether to add points at random      

    #x = np.array(x) #our star vectors

    list1 = [i for i in range(t)]

    ##assesing the density

    nu = 0 #the numerator
    de = 0 #the denominator 
    
    for c in combinations(list1, d):
      dete = abs(np.linalg.det(np.array([x[i] for i in c])))
      nu += dete
      de += dete**2
      
    den = nu/de #the density

    ##getting m

    si = (number/den)**(1/d) #the initial side of the square
    factor = si/k #the scaling factor
    m = math.ceil(2*si*np.sqrt(d)/t)  #changed
    #print("The m value", m)

    ##the rest

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
    
    if(z=="YES"):
      if y>number:
        listquasi2 = listquasi2[:number]

      else:
        for i in range(number - y):
          listquasi2.append([random.uniform(-k/2, k/2) for u in range(d)])

    
    return listquasi2



################read from a file

def read_file(filename):
  #filename - the name of the input file
  x = []
  with open(file) as f:
    line = f.readline()
    while line:
      x.appen([float(i) for i in line.split()])
      line = f.readline
    x = np.array(x)
  return x

################write to a file

def write_file(lista, filename):
  # lista - the list of points
  #filename - the name of the output file
  with open(filename, "w") as f:
    for p in lista:
      s = ""
      for t in p:
        s += str(t)
        s += " "
      f.write(s + '\n')






  
        























 
