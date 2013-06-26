import scipy as sp
from lattice import Lattice
from sys import stdout

length = 512
fps = 20

from cv import FOURCC
from cv2 import VideoWriter
movieWriter = VideoWriter('freeze.avi', FOURCC('U','2','6','3'),
                          fps, (length, length), False)

lattice = Lattice(sp.random.random(size=(length, length)))
samples = fps*60
x = samples/10
for beta in sp.linspace(.01, 10, samples):
    movieWriter.write((128*lattice.state).astype(sp.uint8))
    lattice.step(beta)
    samples -= 1
    if samples % x == 0:
        stdout.write('-')
        stdout.flush()


