import math
import numpy as np


def Sbarriere(trans):
    ''' Matrice de scattering d'un diffuseur quelconque
    '''
    r = - math.sqrt(1 - trans)
    t = -1j*math.sqrt(trans)
    th = t.conjugate()
    rh = r.conjugate()
    return np.array([[r, 0, 0, 0, t, 0, 0, 0],
           [0, r, 0, 0, 0, t, 0, 0],
           [0, 0, rh, 0, 0, 0, th, 0],
           [0, 0, 0, rh, 0, 0, 0, th],
           [t, 0, 0, 0, r, 0, 0, 0],
           [0, t, 0, 0, 0, r,0, 0],
           [0, 0, th, 0, 0, 0, rh, 0],
           [0, 0, 0, th, 0, 0, 0, rh]], dtype=complex)


def Tbarriere(trans):
    ''' Matrice de transfert d'un diffuseur quelconque
    '''
    r = - 1 * 1j / math.sqrt(trans)
    t = - math.sqrt(1 - trans) * 1j / math.sqrt(trans)
    th = t.conjugate()
    rh = r.conjugate()
    return np.array([[r, 0, 0, 0, t, 0, 0, 0],
                     [0, r, 0, 0, 0, t, 0, 0],
                     [0, 0, rh, 0, 0, 0, th, 0],
                     [0, 0, 0, rh, 0, 0, 0, th],
                     [-t, 0, 0, 0, -r, 0, 0, 0],
                     [0, -t, 0, 0, 0, -r,0, 0],
                     [0, 0, -th, 0, 0, 0, -rh, 0],
                     [0, 0, 0, -th, 0, 0, 0, -rh]], dtype=complex)