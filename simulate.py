#!/usr/bin/env python

import scipy as sp
import os
from config import *
from time import time
from subprocess import check_output

#----------------------------- Settings -------------------------------------
# Number of spins in a side of the grid
side = 64
steps = 200
# All energies normalized to 2J = difference in energy between parallel and
# antiparallel spins
# beta = 2J/kT
beta = 2.25
# Number of spins
N = side*side
# Write buffer size
buffer_size = 4096
make_movie = True
#----------------------------------------------------------------------------

# Seed random number generator
seed = int(time())
sp.random.seed(seed)
# Output filename counter
try:
    counter = 1 + max([int(filename[len(output_prefix):-len(output_suffix)])
                       for filename in os.listdir(output_dir)
                       if filename[:len(output_prefix)] == output_prefix
                       and filename[-len(output_suffix):] == output_suffix])
except Exception as e:
    counter = 1
output_filename = os.path.join(output_dir,
                               (output_prefix+'{:05d}').format(counter))

fd = open(output_filename+output_suffix, 'w', buffer_size)
fd.write('# N={:d}\n'.format(N))
fd.write('# beta={:f}\n'.format(beta))
fd.write('# seed={:d}\n'.format(seed))
fd.write('# commit='+check_output(['git', 'log', '--pretty=format:%H', '-n',
                                   '1' ])+'\n')
grid = sp.empty((side, side), dtype=sp.byte)
# up is 1, down is 0
grid[:,:] = sp.random.random_integers(0, 1, size=grid.shape)
# Look into the possibility of storing (up neighbours - down neighbours) for
# each spin
# It would make the deltaE calculation a simple read, but every spin flip would
# require 4 increments/decrements
# neighbours = sp.zeros(size=grid.shape, dtype=sp.byte)

data = sp.empty((steps, 2), dtype=sp.uint32)
# Number of up spins
up = data[:,0]
up[0] = sp.sum(grid)
# Number of up-down pairs
# Energy = -J * (N++ + N-- - N+-) = 2J * (N+- - N)
updown = data[:,1]
# up ^ up = down ^ down = 0
# up ^ down = down ^ up = 1
# So if I sum the xor of all pairs, I get the number of up-down pairs.
updown[0] = sp.sum(grid[:-1,:] ^ grid[1:,:]) \
          + sp.sum(grid[:,:-1] ^ grid[:,1:]) \
          + sp.sum(grid[-1,:] ^ grid[0,:])   \
          + sp.sum(grid[:,-1] ^ grid[:,0])

if make_movie:
    import cv, cv2
    movieWriter = cv2.VideoWriter(output_filename+'.avi',
                                  cv.FOURCC('U','2','6','3'), 20, (side, side),
                                  False)

for step in xrange(1,steps):
    up[step] = up[step-1]
    updown[step] = updown[step-1]
    for i in range(side):
        for j in range(side):
            # Calculate change in up-down neighbours if I flip out
            deltaupdown = grid[i,(j+1)%side] + grid[i,(j-1)%side] \
                        + grid[(i+1)%side,j] + grid[(i-1)%side,j] - 2
            if not grid[i,j]:
                deltaupdown *= -1
            if deltaupdown <= 0 or \
                    sp.random.random() < sp.exp(-beta*deltaupdown):
                grid[i,j] = not grid[i,j]
                up[step] += [-1, 1][grid[i,j]]
                updown[step] += deltaupdown
    if make_movie:
        movieWriter.write((128*grid).astype(sp.uint8))
    fd.write(' '.join(str(d) for d in data[step, :])+'\n')
fd.close()

