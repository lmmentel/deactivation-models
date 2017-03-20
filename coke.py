
import numpy as np


def coke_model(y, t, k_1, k_2, k_3, N, dtau):
    '''
    Calculates the current right hand side of the ODE for the coke
    formation model by Olsbye et al.

    Args:
        y (array_like) : current values of the A, B, C concentrations
        t (float) : current time value
        k_1 (float) : rate constant for the A -> B reaction
        k_2 (float) : rate constant for the A -> C reaction
        k_3 (float) : rate constant for the B -> C reaction
        N (array_like) : active site density for each spatial point of
            the reactor
        dtau (float) : spatial step in the units of space time

    Returns:
        dydt (array_like) : current value of the derivates of A, B and C
    '''

    # define views through slices
    a = y[::3]
    b = y[1::3]
    c = y[2::3]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    # dadt, dbdt and dcdt are views of the interleaved output vectors in dydt
    dadt = dydt[::3]
    dbdt = dydt[1::3]
    dcdt = dydt[2::3]

    dadt[0] = 0.0
    dadt[1:] = - np.diff(a) / dtau - (k_1 + k_2) * a[1:] * (N[1:] - c[1:])

    dbdt[0] = 0.0
    dbdt[1:] = - np.diff(b) / dtau + (k_1 * a[1:] - k_3 * b[1:]) * (N[1:] - c[1:])

    dcdt[0] = 0.0
    dcdt[1:] = (k_2 * a[1:] + k_3 * b[1:]) * (N[1:] - c[1:])

    return dydt
