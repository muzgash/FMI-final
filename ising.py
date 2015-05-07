#!/usr/bin/env python

#inclusiones matematicas
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm

#inclusiones graficas
import matplotlib as mpl
import matplotlib.pyplot as plt

#inclusiones del sistema
import sys
import time

if __name__=="__main__":
    N=20
    J=1
    T_list=[0.1,0.5,1]
    pasos=100000
    igrid=np.random.randint(2,size=(N,N))*2-1

    allEvH=[]
    allMvH=[]
    for kT in T_list:
        EvH=np.array([])
        MvH=np.array([])
        for H in np.linspace(-1,1,100):
            grid=igrid.copy()
            #Calcular la energia y la magnetizacion
            M=grid.mean()
            E=-M
            for i in range(N):
                for j in range(N):
                    E-=0.5*J*grid[i%N,j%N]*(grid[(i+1)%N,j%N]+grid[(i-1)%N,j%N]+grid[i%N,(j+1)%N]+grid[i%N,(j-1)%N])
            #Pasos montecarlo
            for t in range(pasos):
                p1,p2=np.random.randint(N,size=2);
                #Hallamos el delta de energia
                dE=grid[p1,p2]*(J*(grid[(p1+1)%N,p2%N]+grid[(p1-1)%N,p2%N]+grid[p1%N,(p2+1)%N]+grid[p1%N,(p2-1)%N])+H)
                #Hallamos el delta de la magnetizacion
                dM=-2*grid[p1,p2]
                #Escogemos de manera probabilistica si nos movemos al nuevo estado
                if dE<0:#Si E decrece nos quedamos definitivamente con el estado
                    grid[p1,p2]*=-1
                    E+=dE
                    M+=dM
                else: #si E no decrece
                    r=np.random.random()
                    p=np.exp(-dE/kT)
                    if r<p:#Existe una posibiidad de cambiar el estado dada por dE y la temperatura
                        grid[p1,p2]*=-1
                        E+=dE
                        M+=dM
                #Fin del condicional de probabilidad
            #Fin del bucle de montecarlo

            #concatenamos la informacion en un arreglo
            EvH=np.append(EvH,E)
            MvH=np.append(MvH,M)
        #Fin del bucle de temperatura
        allEvH.append(EvH)
        allMvH.append(MvH)
    #Fin del bucle de H
    #Graficamos
    H=np.linspace(-1,1,100)
    fig=plt.figure()
    fig.add_subplot(211)
    plt.plot(H/0.1,allEvH[0]/float(N*N),'ro',label='kT=0.1')
    plt.plot(H/0.5,allEvH[1]/float(N*N),'gv',label='kT=0.5')
    plt.plot(H,allEvH[2]/float(N*N),'bs',label='kT=1')
    plt.legend(loc='upper left')
    plt.xlabel('Campo Externo(H)')
    plt.ylabel('Energia')
    fig.add_subplot(212)
    plt.plot(H/0.1,allMvH[0]/float(N*N),'ro',label='kT=0.1')
    plt.plot(H/0.5,allMvH[1]/float(N*N),'gv',label='kT=0.5')
    plt.plot(H,allMvH[2]/float(N*N),'bs',label='kT=1')
    plt.legend(loc='upper left')
    plt.xlabel('Campo Externo(H)')
    plt.ylabel('Magnetizacion')
    ht=np.linspace(-1,1,100)
    plt.plot(ht,np.tanh(ht*10))
    plt.savefig("EeMvsH.pdf",dpi=700)
    plt.savefig("EeMvsH.jpg",dpi=700)
    fig.show()


