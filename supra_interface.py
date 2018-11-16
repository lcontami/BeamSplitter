import cmath
import math
import numpy as np
from splitter import splitter_eh_asym


def AndreevR_allE(x):
    '''
    piecewise Andreev reflexion for ensuring proper Real values
    For now we do not use it
    '''
    if x**2 <= 1:
        return x - 1j * math.sqrt(1 - x**2)
    elif x**2 > 1:
        return x - math.copysign(1, x) * math.sqrt(x**2 - 1)


def AndreevR(x):
    return cmath.exp(-1j * math.acos(x))


def Ssupra(E, Delta):
    '''
    Matrice de scattering pour une interface normal / supraconducteur idéale,
    et deux canaux de conduction; gives a 4x4 matrix

    base : e up (contact 1), e down (contact 1), h up (contact 1),
    h down (contact 1)
    '''
    # structure en spin
    antidiag = np.array([[0, AndreevR(E / Delta)],
                         [AndreevR(E / Delta), 0]])
    # structure en e-h
    return np.block([[np.zeros((2, 2)), antidiag],
                     [antidiag, np.zeros((2, 2))]], dtype=complex)


def S_Splitter_Supra(E, Delta, phi):
    ''' Scattering matrix for a three terminal splitter with a superconducting
    contact one one arm  (the one labelled 3)
    In the end, this scattering matrix relates incoming electrons in arm 1 and
    2 with outgoing electrons in the same arm
    The output matrix is a 8x8 matrix base :
    e up (contact 1), e down (contact 1), h up (contact 1), h down (contact 1)
    + 2nd contact
    '''
    # one needs to express the splitter scattering matrix in the full space:
    # 2 spins and e-h:
    # gives a 12x12 matrix:  2 spins x 3 terminals x e-h (2)
    # for now there is no energy dependance for the splitter scattering matrix
    _Splitter = splitter_eh_asym(phi)
    _S33 = _Splitter[8:, 8:]
    _S13 = _Splitter[:4, 8:]
    _S31 = _Splitter[8:, :4]
    _S23 = _Splitter[4:8, 8:]
    _S32 = _Splitter[8:, 4:8]
    A = (np.linalg.inv(np.identity(4) - Ssupra(E, Delta) @ _S33) @
         Ssupra(E, Delta))
    _Sblock = np.block([[_S13 @ A @ _S31, _S13 @ A @ _S32],
                        [_S23 @ A @ _S31, _S23 @ A @ _S32]], dtype=complex)
    return _Splitter[:8, :8] + _Sblock


def S_to_T_8x8(matrix):
    _M11 = matrix[:4, :4]
    _M12 = matrix[:4, 4:]
    _M21 = matrix[4:, :4]
    _M22 = matrix[4:, 4:]
    prod =  _M11 @ np.linalg.inv(_M21)
    return np.block([[_M12 - prod @  _M22, prod],
                     [-np.linalg.inv(_M21) @ _M22, np.linalg.inv(_M21)]],
                     dtype=complex)


def T_Splitter_Supra(E, Delta, phi):
    '''
    Corresponding transfer matrix (8x8 matrix)
    '''
    _Splitter = S_Splitter_Supra(E, Delta, phi)
    return S_to_T_8x8(_Splitter)