from barriers import Tbarriere
from full_tranfer_matrices import segment_propagation_eh, helix_beamS_helix

# now what do we want to look at ?
# the conductance is directly the transmission probability traced over each
# channel


def T_confined_segment(mu, E, h, theta, l, transL, transR):
    ''' Transfer matrix of a segment with the spin helix between two barriers
    '''
    return (Tbarriere(transR) @ segment_propagation_eh(mu, E, h, theta, l) @
            Tbarriere(transL))


def trans_segment(mu, E, h, theta, l, transL, transR):
    ''' Transmission of a segment with the spin helix between two barriers
    '''
    Tmat = (Tbarriere(transR) @ segment_propagation_eh(mu, E, h, theta, l) @
            Tbarriere(transL))
    # la transmission est donné par les elements non diagonaux de la matrice de
    # scattering
    # de plus on a S[2,1] = 1/T[2,2] (où 1,2 sont les indices des contacts)
    T22 = Tmat[:4, :4]
    transmission = np.linalg.inv(T22)
    return np.trace(np.transpose(np.conjugate(transmission)) @ transmission)


def trans_segment_fast(mu, E, h, theta, l, transL, transR):
    ''' Transmission of a segment with the spin helix between two barriers,
        faster calculation
    '''
    # on n'a pas besoin de calculer la matrice de transfert totale
    Tcenter = segment_propagation_eh(mu, E, h, theta, l)
    T22 = (Tcenter[:4, :4] * math.sqrt(1-transL) -
           Tcenter[4:, 4:] * math.sqrt(1-transR) +
           Tcenter[4:, :4] * math.sqrt(1-transL) -
           Tcenter[:4, 4:]) / math.sqrt(transL * transR)
    S21 = np.linalg.inv(T22)
    return np.trace(np.transpose(np.conjugate(S21)) @ S21)


def trans_helix_beamS_helix(E, Delta, phi, muL, muR, theta_L, theta_R, hL, hR,
                            lL, lR, transL, transR):
    ''' Transmission of the beam splitter with two spin helix + a
        superconducting contact, between two barriers
    '''
    Tmat = helix_beamS_helix(E, Delta, phi, muL, muR, theta_L, theta_R, hL, hR,
                             lL, lR, transL, transR)
    trans = np.linalg.inv(Tmat[:4, :4])
    return np.trace(np.transpose(np.conjugate(trans)) @ trans)