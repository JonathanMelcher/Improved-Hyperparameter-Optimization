#This program generates figure1 from the paper, which compares different
#sampling strategies

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations, product
from matplotlib import colors
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

seed = 424

random.seed(seed)
np.random.seed(13)

plt.style.use('classic')

################generating hyperuniform

def make_hyper(k, number, s, run):
    #number - expected number of points
    #k - the length of the squares
    #s  - stealth
    #run - corresponds to the hyperuniform point file used

    x = []
    if s==0.38 and run>=20:
     o = r"stealthy-point-patterns\stealthy-"+str(s)+r"-1000\stealthy-"+str(s)+r"-lbfgs-1000-run-"+str(run)+".dat"
    else:
     o = r"stealthy-point-patterns\stealthy-"+str(s)+r"-1000\stealthy-"+str(s)+r"-lbfgs-1000-run-"+str(run)+".txt"
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

################making a grid

def make_grid(number, k, d):
 #k - the length of the square
 #d - the number of dimensions
 #number - expected number of points
    
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

################making a random point pattern

def make_random(number, k, d):
    #number - expected number of points
    #k - the length of the squares
    #d - the number of dimensions
    listrand = []
    for r in range(number):
      listrand.append([random.uniform(-k/2, k/2) for j in range(d)])
    return listrand


################transforming a point pattern to new boundaries


def transform_tiling(tiling, old_min, old_max, new_min, new_max, d):
    #d is the number of dimensions
    #old_min is the vector with the old minimum, old_max - maximum
    #new_min is the new minimum, new_max - maximum
    for i in range(d):
        a = (new_max[i]-new_min[i])/(old_max[i]-old_min[i])
        b = (new_min[i]*old_max[i]-old_min[i]*new_max[i])/(old_max[i]-old_min[i])
        for point in tiling:
            point[i] = a*point[i]+b


################plotting a point pattern

def plotting2(lista, t, axs, size=5):

  xcoord = []
  ycoord = []

  for i in range(len(lista)):
    xcoord.append((lista[i])[0])
    ycoord.append((lista[i])[1])
    
  if t=='red':
      axs.plot(xcoord, ycoord, color='red', marker = 'o', linestyle = '', markersize=size, linewidth=2)
      return
  if t=='darkgreen':
      axs.plot(xcoord, ycoord, color='darkgreen', marker = 'o', linestyle = '', markersize=size, linewidth=2)
      return
  axs.plot(xcoord, ycoord, t, markersize=size, linewidth=2)

  
     

d=2

x0 = 1-7/9
y0 = 0.4

def g2(x):
    return np.exp(-abs(x[0]-x0)**0.7-0.03*abs(x[1]-y0)**0.7)

def g(x):
    return np.exp(-abs(x-x0)**0.7)

def f(x, point, A, w):
    #x - the vector
    #A - the matrix making a linear transformation of the coordinates
    #w - the vector of weights
    #p - the point from which we calculate the distance

    return w*np.linalg.norm(np.array([x[i] - point[i] for i in range(d)]))**0.3


list_A = []
list_w = []
list_point = []



for i in range(5):
    A = [[1 for q in range(d)] for r in range(d)]
    A = A/np.linalg.norm(A)
    w = np.random.uniform(-5, 5)
    point = np.array([np.random.uniform(-1, 1) for i in range(d)])
    list_A.append(A)
    list_w.append(w)
    list_point.append(point)



list_w[0]=9
list_point[-1] = np.array([0.31412189, -0.5429194])

def f_total(x):
    return np.sum([f(x, list_point[i], list_A[i], list_w[i]) for i in range(5)])


X = np.arange(-1, 1.01, 0.01)
num = len(X)
Y = np.arange(-1, 1.01, 0.01)
X, Y = np.meshgrid(X, Y)

Z = np.array([[f_total([X[i, j], Y[i, j]]) for j in range(num)] for i in range(num)])


#function making each panel of the figure

def do_figure(ax, axs, l, size=2, numlevels=7, num_dashes=20):
    #axs is the part with the projection
    #ax is the 2D part with the point pattern

    Z = np.array([[g2([X[i, j], Y[i, j]]) for j in range(num)] for i in range(num)])

    if num_dashes>len(l):
      num_dashes=len(l)

    list_dashes = random.sample(l, num_dashes)

    X2 = np.arange(-1, 1.01, 0.01)

    l2 = [[X2[i], g(X2[i])] for i in range(len(X2))]
    plotting2(l2, 'k-', axs, size=1)

    l3 = [g(X2[i]) for i in range(len(X2))]
    indeks_maks = l3.index(max(l3))

    hit_maximum = 0

    list_darkgreen = []
    
    for p in l:
      if g2(p)>=0.8:
        list_darkgreen.append(p)
        hit_maximum=1

    if hit_maximum==1:
        color_maximum='darkgreen'
    else:
        color_maximum='red'

    cmap=cm.Blues.reversed()

    colorlist=[]

    for i in np.linspace(0.2, 1, numlevels+1):
      colorlist.append(cmap(i))

    colorlist[0]=color_maximum
    colorlist=colorlist[:-2]

    cmap=colors.ListedColormap(colorlist)
    cmap=cmap.reversed()

    colour1=ax.contourf(X, Y, Z, levels = [0.3, 0.4, 0.5, 0.65, 0.8, 1], cmap = cmap, extend = 'both')
    colour2=ax.contour(X, Y, Z, levels = [0.4, 0.5, 0.65, 0.8, 1], colors='k')    

    #drawing the dashed lines

    for i in range(0, r_grid*2):
        min_y = -k/2
        if_found = 0
        chosen = []
        for p in l:
            if p[0]>=-k/2+(k/(2*r_grid))*i and p[0]<=-k/2+(k/(2*r_grid))*(i+1) and p[1]>min_y:
                if_found=1
                chosen = p
                min_y=p[1]
        if if_found:
            
            #plotting the dashed lines in the upper part
            axs.plot([chosen[0], chosen[0]], [0.9*min(l3), g(chosen[0])], 'k--', linewidth=2)

            #plotting the dashed lines in the lower part
            ax.plot([chosen[0], chosen[0]], [chosen[1], k/2], 'k--', linewidth=2)
               
    
    plotting2(l, 'red', ax, size)

    for p in list_darkgreen:
        ax.plot(p[0], p[1], color='darkgreen', marker='o', markersize=size)

    
    m = [[l[i][0], g(l[i][0])] for i in range(len(l))]
    plotting2(m, 'red', axs, size)

    for p in list_darkgreen:
        axs.plot(p[0], g(p[0]), color='darkgreen', marker='o', markersize=size)


    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)

    
    l3 = [g(X2[i]) for i in range(len(X2))]
    axs.set_xlim(-k/2, k/2)
    axs.set_ylim(0.9*min(l3), 1.1*max(l3))
    axs.set_aspect(0.5*k/(1.1*max(l3) - 0.9*min(l3)))
    

    axs.xaxis.set_ticklabels([])
    axs.yaxis.set_ticklabels([])

    for line in axs.xaxis.get_ticklines():
        line.set_visible(False)
    for line in axs.yaxis.get_ticklines():
        line.set_visible(False)


def do_figure_2(ax, l, size=2, numlevels=7):
    #axs is the part with the projection
    #ax is the 2D part with the tiling

    Z = np.array([[f_total([X[i, j], Y[i, j]]) for j in range(num)] for i in range(num)])    

    color_maximum = 'red' #indicator if the extremum was sampled
    
    cmap=cm.Blues.reversed()
    colorlist=[]

    for i in np.linspace(0.2, 1, 13):
      colorlist.append(cmap(i))

    colorlist[0]='red'
    colorlist = [colorlist[0]] + colorlist[3:]

    cmap=colors.ListedColormap(colorlist)

    colour2=ax.contour(X, Y, Z, levels = [10.5, 12, 13.5, 15, 18], colors='k')
    colour1=ax.contourf(X, Y, Z, levels = [10.5, 12, 13.5, 15, 18], cmap = cmap, extend='both')

    list_darkgreen = []
    
    for p in l:
      if f_total(p)<=colour2.levels[0]:
        list_darkgreen.append(p)
        color_maximum = 'darkgreen'

    if color_maximum=='darkgreen':
        colorlist[0]='darkgreen'
        cmap=colors.ListedColormap(colorlist)
        colour1=ax.contourf(X, Y, Z, levels = [10.5, 12, 13.5, 15, 18], cmap = cmap, extend='both')
        

    plotting2(l, 'red', ax, size)

    
    for p in list_darkgreen:
        ax.plot(p[0], p[1], color='darkgreen', marker='o', markersize=size)


    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)


x = 3
wspace = 0.1
hspace_2 = 0.2
fig = plt.figure(figsize = (x*(3+2*wspace), x*(2.5+hspace_2)))
size = 8

gs_all = GridSpec(2, 1, hspace = hspace_2, wspace = wspace, height_ratios = [3, 2])
gs = GridSpecFromSubplotSpec(12, 6, subplot_spec = gs_all[0], hspace = 0, wspace = wspace)
gs2 = GridSpecFromSubplotSpec(8, 6, subplot_spec = gs_all[1], hspace = 0, wspace = wspace)

ax_down_1 = fig.add_subplot(gs[4:12, 0:2])
ax_up_1 = fig.add_subplot(gs[0:4, 0:2])
ax_down_down_1 = fig.add_subplot(gs2[:, 0:2])
ax_down_1.set_aspect(1)
ax_down_down_1.set_aspect(1)

ax_down_2 = fig.add_subplot(gs[4:12, 2:4])
ax_up_2 = fig.add_subplot(gs[0:4, 2:4])
ax_down_down_2 = fig.add_subplot(gs2[:, 2:4])
ax_down_2.set_aspect(1)
ax_down_down_2.set_aspect(1)

ax_down_3 = fig.add_subplot(gs[4:12, 4:6])
ax_up_3 = fig.add_subplot(gs[0:4, 4:6])
ax_down_down_3 = fig.add_subplot(gs2[:, 4:6])
ax_down_3.set_aspect(1)
ax_down_down_3.set_aspect(1)




d=2
k=2
number = 25
r_grid = 5

listhyper = make_hyper(k, number, 0.49, 20)

list_grid = make_grid(number, k, d)
list_random = make_random(number, k, d)

do_figure(ax_down_2, ax_up_2, listhyper, size)
ax_up_2.set_title('Hyperuniform')
ax_down_2.set_title("One variable dominant: Hyperuniform and Random good in projection", y=-0.15)
do_figure(ax_down_1, ax_up_1, list_grid, size)
ax_up_1.set_title('Grid')
do_figure(ax_down_3, ax_up_3, list_random, size)
ax_up_3.set_title('Random')

do_figure_2(ax_down_down_1, list_grid, size)
do_figure_2(ax_down_down_2, listhyper, size)
do_figure_2(ax_down_down_3, list_random, size)
ax_down_down_2.set_title("Both variables important: Hyperuniform and Grid more isotropic", y=-0.15)


plt.savefig("Comparison.png", bbox_inches='tight', dpi=600)
plt.show()



