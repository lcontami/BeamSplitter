import numpy as np
import math
import cmath
import os
from functools import lru_cache


# A nouveau, la base choisie pour les matrices de scattering qui suivent est la suivante:
# e up (contact 1), e down (contact 1), e up (contact 2), e down (contact 2)
# c'est à dire 2 spins x deux terminaux
# pour l'instant, on a pas ajouté le secteur trous

sigmaz = np.array([[1, 0], [0, -1]])
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigma0 = np.array([[1, 0], [0, 1]])


def kp(h, mu, E):
    return cmath.sqrt(E + mu + h)


def km(h, mu, E):
    return cmath.sqrt(E + mu - h)


def R0(h, mu, E, theta):
    t = theta / 2.
    kps = cmath.sqrt(kp(h, mu, E))
    kms = cmath.sqrt(km(h, mu, E))
    c = math.cos(t)
    s = math.sin(t)
    return np.array([[c / kps, -s / kms, c / kps, -s / kms],
                     [s / kps, c / kms, s / kps, c / kms],
                     [c * kps, -s * kms, -c * kps, s * kms],
                     [s * kps, c * kms, -s * kps, -c * kms]])


def propagation(h, mu, E, L):

    _M = np.array([[cmath.exp(1j * kp(h, mu, E) * L), 0, 0, 0],
                   [0, cmath.exp(1j * km(h, mu, E) * L), 0, 0],
                   [0, 0, cmath.exp(-1j * kp(h, mu, E) * L), 0],
                   [0, 0, 0, cmath.exp(-1j * km(h, mu, E) * L)]])
    return _M


def prod_tau_intermediate(h, mu, E, theta, L):
    # evaluates in 7.82mus with np
    # evaluates in 5.71mus with tn

    # c'est le log de ce que Takis appelle Tau(L) dans son nb:
    # (iK + alpha A / 4 sqrt(kup * kdown) ) L
    A = np.array([[1j * kp(h, mu, E), 0, 0, 0],
                  [0, 1j * km(h, mu, E), 0, 0],
                  [0, 0, -1j * kp(h, mu, E), 0],
                  [0, 0, 0, -1j * km(h, mu, E)]])

    B = np.array([[0, 1, 0, 1],
                  [-1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, -1, 0]]) * kp(h, mu, E) + \
        np.array([[0, 1, 0, -1],
                  [-1, 0, -1, 0],
                  [0, -1, 0, 1],
                  [-1, 0, -1, 0]]) * km(h, mu, E)

    c = theta * 1 / (4 * cmath.sqrt(kp(h, mu, E) * km(h, mu, E)))

    # total function runs in
    # np : 30.1us, same with math.sqrt
    # tn : 23.2us, 14.1us with math.sqrt, 5.02us with math.sqrt in k

    return (A + c * B) * L

# using np for intermediate matrix, scipy.linalg,expm lasts 392us
# this method with ta and math.sqrt takes 97us


def prod_tau(h, mu, E, theta, L):
    # manual exponentiation of the matrix to obtain tau(L)
    d, Y = np.linalg.eig(prod_tau_intermediate(h, mu, E, theta, L))
    Yinv = np.linalg.inv(Y)
    D = np.diag(np.exp(d))

    return np.dot(Y, np.dot(D, Yinv))


def tau(h, mu, E, theta, L, mu_L, mu_R, h_pol, Ll, Lr):
    # ajoute les bords (transition vers de zones avec des amplitudes de champ différent) +
    # segments de propagation de longueurs Ll et Lr
    return propagation(h_pol, mu_L, E, Ll) @ \
        np.linalg.inv(R0(h_pol, mu_L, E, 0)) @ \
        R0(h, mu, E, 0) @ prod_tau(h, mu, E, theta, L) @ \
        np.linalg.inv(R0(h, mu, E, 0)) @ \
        R0(-h_pol, mu_R, E, 0) @ \
        propagation(-h_pol, mu_R, E, Lr)
# changé l'axe pour l'aimantation du dernier domaine
        # np.linalg.inv(R0(h, mu, E, theta * L)


@lru_cache()
def S(h, mu, E, theta, L, mu_L, mu_R, h_pol, Ll, Lr):
    _tau = tau(h, mu, E, theta, L, mu_L, mu_R, h_pol, Ll, Lr)
    # intermediary block matrices of S
#     _tau22_inv = np.linalg.inv(_tau[2:,2:])
    _tau22_inv = np.linalg.inv(_tau[2:, 2:])
    S11 = _tau[:2, 2:] @ _tau22_inv
    S12 = _tau[:2, :2] - S11 @ _tau[2:, :2]
    S21 = _tau22_inv
    S22 = - _tau22_inv @ _tau[2:, :2]

    _S = np.array(np.block([[S11, S12], [S21, S22]]))
    return _S


@lru_cache()
def Sref(h, mu, E, theta, L, mu_L, mu_R, Lw, phi, h_pol, Ll, Lr):
    '''
    Matrice de reflexion, en mettant un mur d'un côté de la structure
    dim 2x2
    '''
    Scattering = S(h, mu, E, theta, L, mu_L, mu_R, h_pol, Ll, Lr)
    S11 = Scattering[:2,:2]
    S12 = Scattering[:2,2:]
    S21 = Scattering[2:,:2]
    S22 = Scattering[2:,2:]
    exp_phi = cmath.exp(1j * (phi + 2 * kp(0, mu, E) * Lw))


    _S = S11 + S12 @ \
        np.linalg.inv((np.identity(2) - exp_phi * S22)) @ \
        S21 * exp_phi

    return _S
