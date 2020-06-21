import numpy as np
from numpy import linalg as LA


def shmidt_decomp(f, modes_num=40):
    """
    :param f: Complex function of two arguments as 2D array.
    :param modes_num: Number of modes in the summ
    :return: modes, eigenvalsues, phases
    """

    grd = len(f)

    dm1 = np.zeros((grd, grd), dtype=complex)
    dm2 = np.zeros((grd, grd), dtype=complex)

    for i in range(grd):
        for j in range(grd):
            dm1[i, j] = np.trapz(np.multiply(f[i, :], np.conj(f[j, :])))
            dm2[i, j] = np.trapz(np.multiply(f[:, i], np.conj(f[:, j])))

    w1, v1 = LA.eig(dm1)
    w2, v2 = LA.eig(dm2)

    eigvals = w1 / np.sum(w1)
    eigvals = np.abs(eigvals)

    modes1 = np.zeros((modes_num, grd), dtype=complex)
    modes2 = np.zeros((modes_num, grd), dtype=complex)

    for n in range(modes_num):
        modes1[n, :] = v1[:, n] / np.sqrt(np.trapz(np.power(np.abs(v1[:, n]), 2)))
        modes2[n, :] = v2[:, n] / np.sqrt(np.trapz(np.power(np.abs(v2[:, n]), 2)))

    phases = np.zeros(modes_num, dtype=complex)

    for n in range(modes_num):
        prod = np.tensordot(modes1[n, :], modes2[n, :], axes=0)
        phases[n] = np.conjugate(np.trapz(np.trapz(np.multiply(np.conj(f), prod))) / np.sqrt(eigvals[n]))

    return modes1, modes2, eigvals, phases
