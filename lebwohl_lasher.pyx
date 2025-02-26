
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, exp
from cython.parallel import prange

#=======================================================================
cpdef double one_energy(double[:, :] arr, int ix, int iy, int nmax) nogil:
    """
    Compute the energy of a single lattice cell with periodic boundaries.
    """
    cdef double en = 0.0
    cdef int ixp = (ix + 1) % nmax
    cdef int ixm = (ix - 1) % nmax
    cdef int iyp = (iy + 1) % nmax
    cdef int iym = (iy - 1) % nmax
    cdef double ang

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    return en


#=======================================================================
def all_energy(cnp.ndarray[double, ndim=2] arr, int nmax):
    """
    Compute the total energy of the lattice.
    """
    cdef double enall = 0.0
    cdef int i, j

    # Use memoryview for performance
    cdef double[:, :] arr_view = arr

    # OpenMP parallel execution
    for i in prange(nmax, nogil=True):
        for j in range(nmax):
            enall += one_energy(arr_view, i, j, nmax)

    return enall  

#=======================================================================
def MC_step(cnp.ndarray[double, ndim=2] arr, double Ts, int nmax):
    """
    Perform a Monte Carlo step with Metropolis-Hastings algorithm.
    """
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz, accept = 0
    cdef double scale = 0.1 + Ts

    # Use memoryview
    cdef double[:, :] arr_view = arr

    cdef cnp.ndarray[int, ndim=2] xran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=np.int32)
    cdef cnp.ndarray[int, ndim=2] yran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=np.int32)
    cdef cnp.ndarray[double, ndim=2] aran = np.random.normal(scale=scale, size=(nmax, nmax))

    for i in prange(nmax, nogil=True):  # Parallel execution
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            en0 = one_energy(arr_view, ix, iy, nmax)  
            arr_view[ix, iy] += ang
            en1 = one_energy(arr_view, ix, iy, nmax)  

            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                
                # Re-acquire GIL only for this Python function
                with gil:
                    if boltz >= np.random.uniform(0.0, 1.0):
                        accept += 1
                    else:
                        arr_view[ix, iy] -= ang

    return accept / (nmax * nmax)


