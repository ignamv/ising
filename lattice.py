#--encoding: utf-8 --
import scipy as sp

class Lattice(object):
    """Grid of dipoles with nearest-neighbour interactions

    2D grid of dipoles which can either point up or down. Every pair of
    anti-parallel neighbours adds one unit of energy. On every simulation step,
    the lattice evolves by always minimizing energy and ocasionally (depending
    on the temperature) gaining energy."""


    def __init__(self, state):
        """Initialize the lattice to the specified state (2D array of 1,0)"""
        if state.shape[0] % 2 != 0 or state.shape[1] % 2 != 0:
            raise Exception('Lattice side lengths must be even')
        self.state = state.astype(sp.byte)
        # Stores info on the neighbours of the slice I'm updating
        self.dUpdown = sp.empty((self.state.shape[0]/2, self.state.shape[1]/2),
                                dtype=sp.byte)
        self.updateUp()
        self.updateUpDown()

    def updateUp(self):
        self._up = sp.sum(self.state)

    def updateUpDown(self):
        # up ^ up = down ^ down = 0
        # up ^ down = down ^ up = 1
        # So if I sum the xor of all pairs, I get the number of up-down pairs.
        self._updown = sp.sum(self.state[:-1,:] ^ self.state[1:,:]) \
                     + sp.sum(self.state[:,:-1] ^ self.state[:,1:]) \
                     + sp.sum(self.state[-1,:] ^ self.state[0,:])   \
                     + sp.sum(self.state[:,-1] ^ self.state[:,0])
    @property
    def up(self):
        """Number of spins pointing up"""
        return self._up

    @property
    def updown(self):
        """Number of up-down neighbours"""
        return self._updown


    def step(self, beta):
        """Simulate one time step with a temperature such that kT = 1/beta

        The unit of energy is that required to make two neighbours antiparallel
        """
        # Update spins by slices
        # This allows me to vectorize the counting of neighbours
        for x,y in sp.random.permutation([(0,0), (1,1), (1,0), (0,1)]):
            # Change in up-down neighbours for each cell I'm updating
            self.dUpdown[:,:] = -2
            # Handle wrapping (different for each slice)
            if x == 0:
                self.dUpdown += self.state[1::2,y::2]
                self.dUpdown[1:,:] += self.state[1:-1:2,y::2]
                self.dUpdown[0,:] += self.state[-1,y::2]
            else:
                self.dUpdown[:-1,:] += self.state[2::2,y::2]
                self.dUpdown[-1,:] += self.state[0,y::2]
                self.dUpdown += self.state[0::2,y::2]
            if y == 0:
                self.dUpdown += self.state[x::2,1::2]
                self.dUpdown[:,1:] += self.state[x::2,1:-1:2]
                self.dUpdown[:,0] += self.state[x::2,-1]
            else:
                self.dUpdown[:,:-1] += self.state[x::2,2::2]
                self.dUpdown[:,-1] += self.state[x::2,0]
                self.dUpdown += self.state[x::2,0::2]
            # Change sign if the cell is pointing down
            self.dUpdown *= self.state[x::2, y::2]*2-1
            # Flip spins with probability min{exp(-ß*ΔE), 1}
            flip = sp.random.random(size=self.dUpdown.shape) \
                 < sp.exp(-beta*self.dUpdown)
            self.state[x::2,y::2] ^= flip
            self._updown += sp.sum(self.dUpdown * flip)
        self.updateUp()

