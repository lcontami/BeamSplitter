import numpy as np
import math
import cmath


def splitter_onechannel_sym(phi):
    ''' Beam splitter for a three terminal scatterer with one conduction channel
       the phi parameter enables to tune from a total reflection to an equal
       splitting, with 3 indistinguishable arms
    '''
    diag = 1. + 2. * cmath.exp(1j*phi)
    other = 1. - cmath.exp(1j*phi)
    return np.array([[diag, other, other],
                     [other, diag, other],
                     [other, other, diag]]) / 3.


def splitter_onechannel_asym(phi):
    ''' Beam splitter for a three terminal scatterer with one conduction channel
        the phi parameter enables one to switch from
        no transmission to the third contact (phi=0) to a symetric case (phi=2)
    '''
    sqroot = math.sqrt(9-2*phi**2)
    diag = 3 - sqroot
    return np.array([[diag, diag-6, 2*phi],
                     [diag-6, diag, 2*phi],
                     [2*phi, 2*phi, 2*(3-diag)]]) * (-1/6.)


def splitter_sym(phi):
    ''' Beam splitter for a three terminal scatterer with two conduction
    channel (two spins) this is a 6x6 matrix
    '''
    diag = np.array([[1. + 2. * cmath.exp(1j*phi), 0],
                     [0, 1. + 2. * cmath.exp(1j*phi)]]) / 3.
    other = np.array([[1. - cmath.exp(1j*phi), 0],
                      [0, 1. - cmath.exp(1j*phi)]]) / 3.
    return np.block([[diag, other, other],
                     [other, diag, other],
                     [other, other, diag]])


def splitter_eh_sym(phi):
    ''' Beam splitter for a three terminal scatterer with two conduction
        channel (two spins) and the e-h subspace
        this is a 12x12 matrix
        base : e up (contact 1), e down (contact 1), h up (contact 1), h down
        (contact 1), idem contact 2, idem contact 3
    '''
    diag = np.array([[1. + 2. * cmath.exp(1j*phi), 0],
                     [0, 1. + 2. * cmath.exp(1j*phi)]]) / 3.
    other = np.array([[1. - cmath.exp(1j*phi), 0],
                      [0, 1. - cmath.exp(1j*phi)]]) / 3.
    diagh = np.conj(diag)
    otherh = np.conj(other)
    zero = np.zeros((2,2))
    return np.block([[diag, zero, other, zero, other, zero],
                     [zero, diagh, zero, otherh, zero, otherh],
                     [other, zero, diag, zero, other, zero],
                     [zero, otherh, zero, diagh, zero, otherh],
                     [other, zero, other, zero, diag, zero],
                     [zero, otherh, zero, otherh, zero, diagh]])


def splitter_eh_asym(phi):
    ''' Beam splitter for a three terminal scatterer with two conduction
        channel (two spins) and the e-h subspace
        this is a 12x12 matrix
        base : e up (contact 1), e down (contact 1), h up (contact 1), h down
        (contact 1), idem contact 2, idem contact 3
    '''
    sqroot = math.sqrt(9-2*phi**2)
    diag = 3 - sqroot

    diag1 = np.array([[3-sqroot, 0],
                     [0, 3-sqroot]])
    diag2 = np.array([[2*(3-diag), 0],
                     [0, 2*(3-diag)]])
    other1 = np.array([[diag-6, 0],
                      [0, diag-6]])
    other2 = np.array([[2*phi, 0],
                      [0, 2*phi]])
    # here there is no complex number
    # diagh = np.conj(diag)
    # otherh = np.conj(other)
    zero = np.zeros((2,2))
    return np.block([[diag1, zero, other1, zero, other2, zero],
                     [zero, diag1, zero, other1, zero, other2],
                     [other1, zero, diag1, zero, other2, zero],
                     [zero, other1, zero, diag1, zero, other2],
                     [other2, zero, other2, zero, diag2, zero],
                     [zero, other2, zero, other2, zero, diag2]]) * (-1/6.)