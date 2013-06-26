Spin lattice simulator
======================

simulate.py simulates a grid of magnetic dipoles (think bar magnets) which can
either point up or down. Adjacent magnets try to become parallel, tending to
regions in the lattice where the direction is constant. Meanwhile, thermal
noise tends to flip dipoles randomly, which can destroy any semblance of order
in the lattice. 

The order of the lattice is measured by the number of spins
pointing up/down and by the number of up-up, up-down and down-down pairs. The
point of this program is to run a number of simulations at various temperatures
and measure the variations in order. Also, to make [pretty
videos](http://www.youtube.com/watch?v=5hcBQ8a1Bag)
Usage
-----

    python simulate.py -h

for usage options


