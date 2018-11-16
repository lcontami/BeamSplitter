import numpy as np
from field_helix import R0, prod_tau
from barriers import Tbarriere
from supra_interface import T_Splitter_Supra


def segment_propagation(mu, E, h, theta, l):
    ''' Transfer matrix for a segment of spin helix, with total angle theta,
    length l, osc field magnitude h,
    for electrons with energy E and chemical potential mu
    '''
    # zone avec un champ tournant h, angle total theta, longueur l
    # on a besoin des matrices de rotation aux bords qui indiquent qu'autour
    # de la texture il n'y a pas de champ magnétique !
    return np.linalg.inv(R0(0, mu, E, 0)) @ \
           R0(h, mu, E, 0) @ prod_tau(h, mu, E, theta, l) @ \
           np.linalg.inv(R0(h, mu, E, 0)) @ \
           R0(0, mu, E, 0)


# Remarque: Takis n'a pas ajouté électrons / trous;
# la matrice finale de beam splitter mélange e et h: il va falloir redéfinir
# la propagation dans la base complète
sigmay = np.array([[0, -1j], [1j, 0]])


def segment_propagation_eh(mu, E, h, theta, l):
    '''
    Complete transfer matrix for the same segment, in the full space:
    e up, e down, h up, h down !
    We get a 8x8 matrix
    '''
    reduced = segment_propagation(mu, E, h, theta, l)
    red_mE = segment_propagation(mu, -E, h, theta, l)
    full = np.zeros((8, 8), dtype=complex)
    for i in [0, 4]:
        for j in [0, 4]:
            full[i:i+2, j:j+2] = reduced[i//2:i//2+2, j//2:j//2+2]
            full[i+2:i+4, j+2:j+4] = np.conj(red_mE[i//2:i//2+2, j//2:j//2+2])
    return full

def helix_beamS_helix_noconf(E, Delta, phi, muL, muR, theta_L, theta_R, hL,
                             hR, lL, lR):
    '''
    Transfer matrix for a beam splitter with two propagating segments on each
    side.
    For simplicity now we put all the segment ends at polarization theta=0 in
    the rotating matrices.
    It would be more rigourous to actually keep track of the correct angle
    8x8 matrix
    '''
    left_prop = segment_propagation_eh(muL, E, hL, theta_L, lL)
    right_prop = segment_propagation_eh(muR, E, hR, theta_R, lR)
    return right_prop @ T_Splitter_Supra(E, Delta, phi) @ left_prop


def helix_beamS_helix(E, Delta, phi, muL, muR, theta_L, theta_R, hL, hR, lL,
                      lR, transL, transR):
    '''
    Transfer matrix for a beam splitter with two propagating segments on each
    side
    We add two barriers of transmission transL and transR on both side to
    enforce confinement.

    For simplicity now we put all the segment ends at polarization theta=0 in
    the rotating matrices.
    It would be more rigourous to actually keep track of the correct angle
    8x8 matrix
    '''
    left_prop = segment_propagation_eh(muL, E, hL, theta_L, lL)
    right_prop = segment_propagation_eh(muR, E, hR, theta_R, lR)
    return (Tbarriere(transR) @ right_prop @ T_Splitter_Supra(E, Delta, phi) @
            left_prop @ Tbarriere(transL))