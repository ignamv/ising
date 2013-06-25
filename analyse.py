#!/usr/bin/env python

import scipy as sp
from sys import argv
from config import *

if len(argv) != 2:
    print 'Usage: ' + argv[0] + ' [run file]'
    exit(-1)

fd = open(argv[1], 'r')
# TODO: cleaner header handling
N = int(fd.readline()[4:-1])
beta = float(fd.readline()[7:-1])
print 'N = {:d}'.format(N)
print 'beta = {:f}'.format(beta)

up, updown = sp.transpose(sp.loadtxt(fd))
energy = 2*(updown - N)

import pylab as plt
plt.figure()
plt.plot(up)
plt.ylabel(r'$N_+$')
plt.xlabel('Step')
plt.savefig(argv[1][:-len(output_suffix)]+'up.png')
plt.figure()
plt.plot(energy)
plt.ylabel('Energy')
plt.xlabel('Step')
plt.savefig(argv[1][:-len(output_suffix)]+'energy.png')

plt.show()
