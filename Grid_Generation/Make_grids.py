
# import libraries
import numpy as np
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt

import GeneratingMultidimensional as gm

def randcoor(d,L):
    coor = []
    for _ in range(d):
        coor.append(random.uniform(0,L))
    return coor

def cheak_overlap(coor, spheres, R):
    for i in spheres:
        i = np.array(i)
        coor = np.array(coor)
        dist = np.sum(np.power(norm(coor-i),2))
        if dist < R**2:
            return False
    return True    

def rand_sphere(d,L,R,N):
    spheres = []
    count = 0
    flag = True
    while len(spheres) < N:
        coor = randcoor(d,L)
        count += 1
        if cheak_overlap(coor, spheres, R):
            spheres.append(coor)
            count = 0
        if count > 100:
            print("too many tries only got", len(spheres), "spheres", end = "\r")
            flag = False
            break
    return spheres, flag


def find_best_R(d,L,N, step = 0.1, R0 = 0.01):
    R = R0
    while rand_sphere(d,L,R,N)[1]:
        R += step

    return R

# define a function to transform the grids into the correct format domain

def transform_grid(grid, prev_min, prev_max, new_min, new_max):
    return (grid - prev_min) / (prev_max - prev_min) * (new_max - new_min) + new_min

def plot_grid(grid, method_name, d = 3):
    #2d plot
    if d == 2:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(grid[:,0], grid[:,1], s=10)
        plt.savefig(method_name + '_grid.png')

    if d == 3:
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        ax.scatter3D(grid[:,0], grid[:,1], grid[:,2], s=10)
        plt.savefig(method_name + '_grid.png')

from itertools import combinations, product


def make_RSA(number, Length, d):
        #number - expected number of points 
    #Length - length of the square

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
    bigvoxel = Length/(Length//D) #big voxel size
    #print("bigvoxel size", bigvoxel)
    v = Length/(Length//(D)) #small voxel initial sizes
    N = int(Length//(D)) #amount of small voxels

    ##for the neighborhood list

    cos = [bigvoxel_amount for i in range(d)]
    n = np.empty(cos, dtype = object)
    n.fill([])
  
    ##checking whether a sphere with center c is good

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

    ##checking whether a voxel with corner c and length v is good

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

    #print("Diameter", D)

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

    ##phase 2 - creating the voxel list
    l = [i*v for i in range(N)]
    for t in product(l, repeat = d):
      if check_voxel(np.array(t), vec, vec_dist):
        vlist.append(np.array(t))
    vnum = len(vlist)

    #print("I have created the voxel list!")

    ##phase 3 - iterating until the voxel list is empty 

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

      if d>=4 and d<=4 and 2**d * len(vlist) > 100000:
        vlist1 = []
        for voxel in vlist:
          if check_voxel(voxel, vec, vec_dist):
            vlist1.append(voxel)
        vlist = list(vlist1)
              
        break

      if 5<=d and d<=8 and 2**d * len(vlist) > 10000:
        vlist1 = []
        #print("starting to check!")
        for voxel in vlist:
          if check_voxel(voxel, vec, vec_dist):
            vlist1.append(voxel)
        vlist = list(vlist1)
              
        #print("Skipping")
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
      #print("ONWARD!")
      #print("WE HAVE THIS MANY", vnum)

    if len(vlist)>=1:
      count_max = max(int(500000/len(vlist)), 30)
    
    #print("Raz", type(vlist))
    random.shuffle(vlist)
    #print("Dwa", type(vlist))

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

def hyper_tilling_grid(resolution, edges, t = 6, k = 1):
    """
    Generate a hyper tilling in the given domain

    Parameters
    ----------
    Resolution: list of integers
        The resolution of the grid in each dimension
    Edges: list of two arrays
        The first array is the lower bound and the second array is the upper bound
    t: integer
        The number of star vectors
    k: integer
        The length of the square
    
    Returns
    -------
    hyper_tilling_transformed: array
        The hyper tilling in the given domain

    Notes
    -----
    The resolution and the two arrays in edges must have the same dimension

    """

    d = len(resolution)
    if d != len(edges[0]):
        raise ValueError("The dimension of the grid and the dimension of the domain do not match")
    
    new_min, new_max = edges

    vectors = gm.make_vectors(d, t)
    gen_tille = gm.make_quasi_nD(np.prod(resolution), k, d,t, vectors, z = 'YES')
    hyper_tilling = np.array(gen_tille)

    ### NOTE: Is this the correct way to do this?
    #         Why is the domain not form -0.5 to 0.5
    old_min = np.ones(d) * np.min(hyper_tilling)
    old_max = np.ones(d) * np.max(hyper_tilling)

    hyper_tilling_transformed = transform_grid(hyper_tilling, old_min, old_max, new_min, new_max)
    return hyper_tilling_transformed



def lhs_grid(resolution, edges):
    """
    Generate a latin hypercube sampling in the given domain

    Parameters
    ----------
    Resolution: list of integers
        The resolution of the grid in each dimension
    Edges: list of two arrays
        The first array is the lower bound and the second array is the upper bound

    Returns
    -------
    lhs_grid: array
        The latin hypercube sampling in the given domain

    Notes
    -----
    The resolution and the two arrays in edges must have the same dimension
    """

    d = len(resolution)
    if d != len(edges[0]):
        raise ValueError("The dimension of the grid and the dimension of the domain do not match")
    
    import pyDOE2
    new_min, new_max = edges

    lhs = pyDOE2.lhs(d, samples = np.prod(resolution))

    old_min = np.zeros(d)
    old_max = np.ones(d)

    lhs_transformed = transform_grid(lhs, old_min, old_max, new_min, new_max)

    return lhs_transformed



def rsa_grid(resolution, edges):
    """
    Generate a random sphere packing in the given domain

    Parameters
    ----------
    Resolution: list of integers
        The resolution of the grid in each dimension
    Edges: list of two arrays
        The first array is the lower bound and the second array is the upper bound
    
    Returns
    -------
    reg_grid: array
        The random sphere packing in the given domain

    Notes
    -----
    The resolution and the two arrays in edges must have the same dimension
    """

    d = len(resolution)
    if d != len(edges[0]):
        raise ValueError("The dimension of the grid and the dimension of the domain do not match")
    
    
    new_min, new_max = edges 

    spheres = np.array(make_RSA(np.prod(resolution), 1, d))
    old_min = np.zeros(d)
    old_max = np.ones(d)

    spheres_transformed = transform_grid(spheres, old_min, old_max, new_min, new_max)
    
    return spheres_transformed



def random_grid(resolution, edges):
    """
    Generate a random grid in the given domain

    Parameters
    ----------
    Resolution: list of integers
        The resolution of the grid in each dimension
    Edges: list of two arrays
        The first array is the lower bound and the second array is the upper bound

    Returns
    -------
    rand_grid: array
        The random grid in the given domain

    """
    d = len(resolution)
    if d != len(edges[0]):
        raise ValueError("The dimension of the grid and the dimension of the domain do not match")
    
    new_min, new_max = edges

    
    temp_grid = []
    for i in range(d):
        temp_grid.append(np.random.uniform(new_min[i], new_max[i], np.prod(resolution)))

    rand_grid = np.vstack((temp_grid)).T

    return rand_grid



def reg_grid(resolution, edges):
    """
    Generate a regular grid in the given domain

    Parameters
    ----------
    Resolution: list of integers
        The resolution of the grid in each dimension
    Edges: list of two arrays
        The first array is the lower bound and the second array is the upper bound
    
    Returns
    -------
    reg_grid: array
        The regular grid in the given domain

    Notes
    -----
    The resolution and the two arrays in edges must have the same dimension
    """

    d = len(resolution)
    if d != len(edges[0]):
        raise ValueError("The dimension of the grid and the dimension of the domain do not match")
    
    new_min, new_max = edges

    temp_grid = []
    for i in range(d):
        temp_grid.append(np.linspace(new_min[i], new_max[i], resolution[i]))
    
    reg_grid = np.meshgrid(*temp_grid)
    reg_grid = np.array(reg_grid).reshape(d,np.prod(resolution)).T

    return reg_grid


if __name__ == '__main__':
    print("This is a module for generating grids for the hyperparameter analysis")
    print("Please import this module and use the functions defined here")
    print("The following functions are available: \n")
    print("hyper_tilling_grid(resolution, edges, t = 6, k = 1)")
    print("lhs_grid(resolution, edges)")
    print("rsa_grid(resolution, edges)")
    print("random_grid(resolution, edges)")
    print("reg_grid(resolution, edges)")
    print("\n " + 80 * "-" + "\n")
    print("""
All functions take two arguments resolution and edges. Resolution is a list of integers 
that defines the resolution of the grid in each dimension. Edges is a list of two arrays
that define the domain of the grid. The first array is the lower bound and the second array
is the upper bound. The dimension of the grid is determined by the length of the resolution list.
Resolution and edges must have the same dimension. \n""")

    resolution = np.array([3, 3, 3])
    edges = [np.array([0.03, 8, 0]), np.array([0.25, 22, 1])]
    resolution = np.array([6, 6])
    edges = [np.array([0.03, 8]), np.array([0.25, 22])]

    print('Test of rsa_grid:')
    rsa_tile_transformed = rsa_grid(resolution, edges)
    plot_grid(rsa_tile_transformed, 'rsa', d = len(resolution))

    print('Test of hyper_tilling_grid:')
    hyper_tilling_transformed = hyper_tilling_grid(resolution, edges)
    plot_grid(hyper_tilling_transformed, 'hyper_tilling', d = len(resolution))

    print('Test of lhs_grid:')
    lhs_tile_transformed = lhs_grid(resolution, edges)
    plot_grid(lhs_tile_transformed, 'lhs', d = len(resolution))

    print('Test of random_grid:')
    rand_grid = random_grid(resolution, edges)
    plot_grid(rand_grid, 'random', d = len(resolution))

    print('Test of reg_grid:')
    reg_grid = reg_grid(resolution, edges)
    plot_grid(reg_grid, 'reg_grid', d = len(resolution))
