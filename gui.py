import argparse
import os
import sys
from gooey import Gooey
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns

from coke import coke_model

nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stdout = nonbuffered_stdout


@Gooey
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('k1', type=float, default=1.0,
                        help='rate constant A --> B')
    parser.add_argument('k2', type=float, default=0.6,
                        help='rate constant A --> C')
    parser.add_argument('k3', type=float, default=0.4,
                        help='rate constant B --> C')
    parser.add_argument('A0', type=float, default=1.0,
                        help='initial concentration of A')
    parser.add_argument('B0', type=float, default=0,
                        help='initial concentration of B')
    parser.add_argument('C0', type=float, default=0,
                        help='initial concentration of C')
    parser.add_argument('Tfinal', type=float, default=5.0,
                        help='final value for time')
    parser.add_argument('Tgrid', type=int, default=50,
                        help='number of grid points for time')
    parser.add_argument('Npoints', type=int, default=100,
                        help='number of points to discretize the reactor')

    cargs = parser.parse_args()

    t = np.linspace(0.0, cargs.Tfinal, num=cargs.Tgrid)
    dtau = 1.0 / cargs.Npoints

    # total number of active sites in [mol / m^3]
    active_sites = 0.82
    print('Active site density: {:.4f}'.format(active_sites))
    # initial values for the number of active sites in each reactor segment
    N0 = np.ones(cargs.Npoints) * active_sites

    # initial values for the concentrations of A, B and C
    abc0 = np.zeros(3 * cargs.Npoints, dtype=float)
    abc0[0:3] = [cargs.A0, cargs.B0, cargs.C0]

    # sovle the ODEs
    abc = odeint(coke_model, abc0, t, args=(cargs.k1, cargs.k2, cargs.k3, N0, dtau))

    sns.set(style='whitegrid')

    plt.figure(1, figsize=(8, 5))
    a = abc[:, ::3]
    b = abc[:, 1::3]

    plt.plot(t, a[:, -1], lw=2.0, label='[A]')
    plt.plot(t, b[:, -1], lw=2.0, label='[B]')
    #plt.plot(t, c[:, -1], lw=2.0, label='[C]')
    plt.legend(loc='upper left')

    plt.title('Outlet concentration vs time')
    plt.xlabel('Time')
    plt.ylabel('Concentration [mol/m$^3$]')

    x = np.linspace(0.0, 1.0, num=cargs.Npoints)
    c = abc[:, 2::3]
    cm = plt.get_cmap('magma_r')
    cNorm = colors.Normalize(vmin=0, vmax=cargs.Tgrid)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    plt.figure(2, figsize=(8, 5))
    for i in range(cargs.Tgrid):
        colorVal = scalarMap.to_rgba(i)
        plt.plot(x, c[i, :], color=colorVal)

    plt.title('Time evolution of coke concentration in the reactor')
    plt.xlabel('Axial reactor coordinate')
    plt.ylabel('Coke concentration [mol/m$^3$]')

    plt.show()


if __name__ == '__main__':

    main()
