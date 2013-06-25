#!/usr/bin/env python
#--encoding: utf-8 --

import scipy as sp
import os
from config import *
from time import time
from subprocess import check_output
import argparse
from sys import stdout

#----------------------------- Settings -------------------------------------
parser = argparse.ArgumentParser(description='Simulate a lattice of dipoles '
                                'with nearest-neighbour interactions. '
                                'The interaction energy is ±1 for '
                                 '[anti]parallel neighbours')
parser.add_argument('-l', '--length', help='Number of lattice sites per side',
                   default=64, type=int)
parser.add_argument('-s', '--steps', help='Number of simulation steps',
                   default=200, type=int)
parser.add_argument('-b', '--beta', help='Inverse temperature, normalized to '
                   'the energy of one spin flip: beta=2/kT', default=2,
                   type=float)
parser.add_argument('-m', '--movie', const=True, default=False, nargs='?',
                    help='Write video to MOVIE. If no filename is '
                    'specified, use OUTPUT with extension set to avi')
parser.add_argument('-o', '--output', help='Data output filename. Default '
                    'value is the next filename of the form '
                    +output_prefix+'NNNNN'+output_suffix)
opt = parser.parse_args()
# Number of spins
N = opt.length*opt.length
# Write buffer size
buffer_size = 4096
#----------------------------------------------------------------------------

# Seed random number generator
seed = int(time())
sp.random.seed(seed)
if opt.output is None:
    try:
        counter = max(int(filename[len(output_prefix):-len(output_suffix)])
                      for filename in os.listdir(output_dir)
                      if filename[:len(output_prefix)] == output_prefix
                      and filename[-len(output_suffix):] == output_suffix)+1
    except Exception as e:
        counter = 1
    opt.output = os.path.join(output_dir, (output_prefix+'{:05d}'+
                                           output_suffix).format(counter))

fd = open(opt.output, 'w', buffer_size)
fd.write('# N={:d}\n'.format(N))
fd.write('# beta={:f}\n'.format(opt.beta))
# Write RNG seed and commit hash so this run can be repeated later, once the
# code has changed
fd.write('# seed={:d}\n'.format(seed))
fd.write('# commit='+check_output(['git', 'log', '--pretty=format:%H', '-n',
                                   '1' ])+'\n')
grid = sp.empty((opt.length, opt.length), dtype=sp.byte)
# up is 1, down is 0
grid[:,:] = sp.random.random_integers(0, 1, size=grid.shape)
# Look into the possibility of storing (up neighbours - down neighbours) for
# each spin
# It would make the deltaE calculation a simple read, but every spin flip would
# require 4 increments/decrements
# neighbours = sp.zeros(size=grid.shape, dtype=sp.byte)

# Original motivation for making up and upup into slices of data was to then
# use data.tofile
# TODO: either use tofile for saving or make them into independent arrays
data = sp.empty((opt.steps, 2), dtype=sp.uint32)
# Number of up spins
up = data[:,0]
up[0] = sp.sum(grid)
# Number of up-down pairs
# Energy = - (N++ + N-- - N+-) = 2 * (N+- - N)
updown = data[:,1]
# up ^ up = down ^ down = 0
# up ^ down = down ^ up = 1
# So if I sum the xor of all pairs, I get the number of up-down pairs.
updown[0] = sp.sum(grid[:-1,:] ^ grid[1:,:]) \
          + sp.sum(grid[:,:-1] ^ grid[:,1:]) \
          + sp.sum(grid[-1,:] ^ grid[0,:])   \
          + sp.sum(grid[:,-1] ^ grid[:,0])

print 'Simulating {:d} steps of a {:d}x{:d} lattice at beta={:f}'.format(
    opt.steps, opt.length, opt.length, opt.beta)
print 'Writing run data to ' + opt.output
if opt.movie:
    if opt.movie != True:
        video_filename = opt.movie
    else:
        if opt.output[-4] == '.':
            video_filename = opt.output[:-3] + 'avi'
        else:
            video_filename = opt.output + '.avi'
    from cv import FOURCC
    from cv2 import VideoWriter
    movieWriter = VideoWriter(video_filename, FOURCC('U','2','6','3'),
                              20, (opt.length, opt.length), False)
    print 'Writing movie to ' + video_filename

# Number of dashes in complete progress bar
progress_bar_size = min(20, opt.steps)
steps_per_dash = opt.steps / progress_bar_size
print '['+' '*progress_bar_size+']\r[',
stdout.flush()
for step in xrange(1,opt.steps):
    up[step] = up[step-1]
    updown[step] = updown[step-1]
    for i in range(opt.length):
        for j in range(opt.length):
            # Calculate change in up-down neighbours if I flip out
            deltaupdown = grid[i,(j+1)%opt.length] + grid[i,(j-1)%opt.length] \
                        + grid[(i+1)%opt.length,j] + grid[(i-1)%opt.length,j] \
                        - 2
            if not grid[i,j]:
                deltaupdown *= -1
            # Transition probability is:
            # - if ΔE < 0: 1
            # - otherwise: exp(-beta*ΔE)
            if deltaupdown <= 0 or \
                    sp.random.random() < sp.exp(-opt.beta*deltaupdown):
                grid[i,j] = not grid[i,j]
                up[step] += [-1, 1][grid[i,j]]
                updown[step] += deltaupdown
    if opt.movie:
        movieWriter.write((128*grid).astype(sp.uint8))
    if step % steps_per_dash == 0:
        stdout.write('-')
        stdout.flush()
if opt.steps % progress_bar_size == 0:
    stdout.write('-')
    stdout.flush()

    fd.write(' '.join(str(d) for d in data[step, :])+'\n')
fd.close()

