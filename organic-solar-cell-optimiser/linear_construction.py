import numpy as np
from numpy import array
import pandas as pd
from os.path import join
from scipy import constants as const
import matplotlib.pyplot as plt


# Units of length are in nano meters (nm).

dataDirectory   = 'data'

layers          = ['SiO2' , 'ITO' , 'PEDOT' , 'P3HTPCBM_BHJ' , 'Ca' , 'Al']

thicknesses     = [100 , 100 , 200 , 500 , 40 , 50]

plotRange       = [300, 800] # Not used


def getData(layers, dataDirectory):
    path = join(dataDirectory,'nk_%s.csv' % layers)
    read = pd.read_csv(path)
    data = array(read)
    return data

for i,mat in enumerate(reversed(layers)):
    if i == 0:
        data = getData(mat, dataDirectory)
    else:
        imp = getData(mat, dataDirectory)
        data = np.append(data, imp[:,1:], axis=1)

#data = getData(layers, dataDirectory)
wl = data[:,0]

def refracIndx(data):
    nk  = data[:, 1:]  # matrix of n and k values
    q   = array([[1+0j]*np.shape(nk)[0]])
    for i in range(0, np.shape(nk)[1], 2):
        q = np.append(q, [nk[:, i] + nk[:, i+1]*1j], axis=0)
    return np.transpose(q)

def Imat(q1, q2):
    r = (q1-q2) / (q1+q2)
    t = (2*q1) / (q1+q2)
    I = np.array([[1/t, r/t], [r/t, 1/t]])
    return I

def Lmat(wl, q1, dj):
    eta = 2*q1*const.pi / wl
    L   = np.array([[np.exp(0-1j*eta*dj), 0],
                     [0, np.exp(0+1j*eta*dj)]])
    return L

q = refracIndx(data)

Rs      = array([])
Ts      = array([])
absorbs = array([])
sysRs   = array([])
sysTs   = array([])
sysAbsorbs = array([])
for i, qi in enumerate(q):
    l   = wl[i]
    stm = Imat(qi[0],qi[1])
    for k in range(1, np.shape(qi)[0]-1):
        dj  = thickness[k-1]
        stm = np.dot(stm, Lmat(l, qi[k], dj))
        stm = np.dot(stm, Imat(qi[k], qi[k+1]))

    R    = np.abs(stm[1,0]/stm[0,0])**2
    T    = np.abs(1/stm[0,0])**2
    incR = ((1-qi[-1]) / (1+qi[-1]))**2
    incT = (4*qi[-1]) / (1+qi[-1])**2
    sysR = incR + (incT**2*R) / (1-incR*R)
    sysT = incT*T / (1-incR*R)
    
    
    Rs          = np.append(Rs, R)
    Ts          = np.append(Ts, T)
    absorbs     = np.append(1-R-T, absorbs)
    sysRs       = np.append(sysRs, sysR)
    sysTs       = np.append(sysTs, sysT)
    sysAbsorbs  = np.append(sysAbsorbs, 1-sysR-sysT)



# Plot the statistics

systemProjection = pd.DataFrame({"Wavelength, nm": wl,
                                "Internal Reflection": sysRs,
                                "Internal Transmision": sysTs,
                                "Absorbtion": sysAbsorbs})

systemPlot = systemProjection.plot("Wavelength, nm", title="Wavelength agains intensity",)