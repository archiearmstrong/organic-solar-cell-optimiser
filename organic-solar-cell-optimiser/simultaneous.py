import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from os.path import join
from scipy import constants as const
import matplotlib.pyplot as plt


# Opens file explorer diaglogue prompting user to select a .csv file.
# Returns selection as a file path to caller.

#def filepath():
#    tk.Tk().withdraw()
#    filewindow = askopenfilename()
#    tk.Tk().destroy()
#    return filewindow

dataDirectory = 'data'
layers = 'original'

# Returns array of complex refraction for atmospheric ambient and metirials
# for energies in the CSV file; the array is constructed as (layers, energies).
def qs(data):
    nk = data[:, 1:]  # matrix of n and k values
    q = np.array([[1+0j]*np.shape(nk)[0]])
    for i in range(0, np.shape(nk)[1], 2):
        q = np.append(q, [nk[:, i] + nk[:, i+1]*1j], axis=0)
    return q


# Returns array of Frenel reflections for each energy
# for the substrate side of layer i.


# Returns
def Imat(q, i):
    r = (q[i]-q[i+1]) / (q[i]+q[i+1])
    t = (2*q[i]) / (q[i]+q[i+1])
    I = np.array([[1/t, r/t], [r/t, 1/t]], dtype=complex)
    return I


def eta(E, q, i):
    wavelength = const.h*const.c / (E*const.e)
    return (2*const.pi / wavelength)*q[i]


def Lmat(E, q, i, dj):
    e = eta(E, q, i)
    return np.array([[np.exp(0-1j*e*dj), 0*e],
                     [0*e, np.exp(0+1j*e*dj)]], dtype=complex)


# Until now array has been in form of (layers, energies).
# To do matrix multiplication the data has to be resorted into matrices.
def S(E, q, dj):
    stm = np.transpose(Imat(q, 0))
    for i in range(1, np.shape(q)[0]-1):
        stm = np.matmul(stm, np.transpose(Lmat(E, q, i, dj)))
        stm = np.matmul(stm, np.transpose(Imat(q, i)))
    return stm


def r(stm):
    r = stm[:, 1, 0] / stm[:, 0, 0]
    return r


def t(stm):
    t = 1 / stm[:, 0, 0]
    return t


def incR(q):
    return ((1-q[-1]) / (1+q[-1]))**2


def incT(q):
    return (4*q[-1]) / (1+q[-1])**2


def sysR(q, R):
    Rstar = incR(q)
    Tstar = incT(q)
    return Rstar + (Tstar**2*R) / (1-Rstar*R)


def sysT(q, R, T):
    Rstar = incR(q)
    Tstar = incT(q)
    return (Tstar*T) / (1-Rstar*R)

path = join(dataDirectory,'%s.csv' % layers)
csvfile = pd.read_csv(path)
data = np.array(csvfile)  # Converts CSV file into numpy array

# Thickness of layer. Given as an array so list of thicknesses can be imported
# for multi-layer structures.
# Like other data, ambiant of the left, substrate on the right.
dj = np.array([100e-9])

# Isolates list of energies
E = data[:, 0]
q = qs(data)
stm = S(E, q, dj)
compref = r(stm)
transco = t(stm)
R = np.absolute(compref)**2
T = np.absolute(transco)**2

# Now accounting for incoherent refraction in the substrate.
systemR = sysR(q, R)
systemT = sysT(q, R, T)

system = pd.DataFrame({"Energy (Ev)": E,
                       "Complex refraction": systemR,
                       "Transsmision coefficient": systemT,
                       "Absorbtion": 1-systemR-systemT})

internal = pd.DataFrame({"Energy (Ev)": E,
                         "Internal Reflection": R,
                         
                         
                         
                         
                         "Internal Transsmision": T,
                         "Absorbtion": 1-R-T})

wavelength = pd.DataFrame({"wavelength (m)": const.h*const.c / (E*const.e),
                         "Internal Reflection": systemR,
                         "Internal Transsmision": systemT,
                         "Absorbtion": 1-systemR-systemT})

systemplot = system.plot("Energy (Ev)", title="Interaction of System regarding Perpendicular Light")
plt.savefig("systemplot.png")
intplot = internal.plot("Energy (Ev)", title="Internal Interaction")
plt.savefig("Internal Interaction")
waveplot = wavelength.plot("wavelength (m)", title="Wavelength agains intensity", xlim=[0.35e-6, 0.8e-6])
plt.savefig("waveplot.png")