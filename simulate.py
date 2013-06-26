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
# Make sure the length is even
opt.length += opt.length % 2
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
# Write RNG seed and commit hash so this run can be repeated after the code has
# changed
fd.write('# seed={:d}\n'.format(seed))
fd.write('# commit='+check_output(['git', 'log', '--pretty=format:%H', '-n',
                                   '1' ])+'\n')
grid = sp.empty((opt.length, opt.length), dtype=sp.byte)
# up is 1, down is 0
grid[:,:] = sp.random.random_integers(0, 1, size=grid.shape)
# Stores the number of neighbours of the slice I'm updating
dUpdown = sp.empty((opt.length/2, opt.length/2), dtype=sp.byte)

# Original motivation for making up and upup into slices of data was to then
# use data.tofile
# TODO: either use tofile for saving or make them into independent arrays
data = sp.empty((opt.steps, 2), dtype=sp.uint32)
# Number of up spins
up = sp.sum(grid)
# Number of up-down pairs
# up ^ up = down ^ down = 0
# up ^ down = down ^ up = 1
# So if I sum the xor of all pairs, I get the number of up-down pairs.
updown = sp.sum(grid[:-1,:] ^ grid[1:,:]) \
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

def evolve_lattice():
    global up, updown, dUpdown
    # Update spins in a chessboard pattern.
    # This allows me to vectorize the counting of neighbours
    for x,y in sp.random.permutation([(0,0), (1,1), (1,0), (0,1)]):
        # Stores the change in the number of up-down pairs if I flip this cell
        dUpdown[:,:] = -2
        # Handle wrapping (different for each slice)
        if x == 0:
            dUpdown += grid[1::2,y::2]
            dUpdown[1:,:] += grid[1:-1:2,y::2]
            dUpdown[0,:] += grid[-1,y::2]
        else:
            dUpdown[:-1,:] += grid[2::2,y::2]
            dUpdown[-1,:] += grid[0,y::2]
            dUpdown += grid[0::2,y::2]
        if y == 0:
            dUpdown += grid[x::2,1::2]
            dUpdown[:,1:] += grid[x::2,1:-1:2]
            dUpdown[:,0] += grid[x::2,-1]
        else:
            dUpdown[:,:-1] += grid[x::2,2::2]
            dUpdown[:,-1] += grid[x::2,0]
            dUpdown += grid[x::2,0::2]
        # Invert if the cell is pointing down
        dUpdown *= grid[x::2, y::2]*2-1
        # Flip with probability min{exp(-ß*ΔE), 1}
        flips = sp.random.random(size=dUpdown.shape) < sp.exp(-opt.beta*dUpdown)
        grid[x::2,y::2] ^= flips
        updown += sp.sum(dUpdown * flips)
    up = sp.sum(grid)

for step in xrange(1,opt.steps):
    evolve_lattice()
    if opt.movie:
        movieWriter.write((128*grid).astype(sp.uint8))
    if step % steps_per_dash == 0:
        stdout.write('-')
        stdout.flush()
    fd.write(' '.join(str(d) for d in [up, updown]) + '\n')
if opt.steps % progress_bar_size == 0:
    stdout.write('-')
    stdout.flush()

fd.close()

