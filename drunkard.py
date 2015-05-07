#!/bin/env python

#inclusiones matematicas
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm

#inclusiones graficas
import matplotlib as mpl
import matplotlib.pyplot as plt

#inclusiones del sistema
import sys



def uso():
    print('Operacion invalida\n')
    print('Uso:\n\tdrunkard(n,N,probailidades)\n')




def uso2():
    print('Operacion invalida\n')
    print('Uso:\n\t2drunkard(n,N,prob1,prob2)\n')
   
   



def drunkard(n,N,probabilidades=(0.25,0.25,0.25,0.25)):
    ''' Muestra un camino y calcula el histograma de frecuencia en funcion
    de la posicion solapada con una funcion gaussiana

    Parameters
    ----------
    n : entero
      Numero de pasos.
    N : entero
      Numero de veces que se pone a caminar al borracho.
    probabilidades : tupla
      Tupla que determina las probabilidades de ir en cada direccion
      Cada valor debe estar entre 0 y 1.
      Se pueden ingresar tuplas de dimension 2,4 y 8.
      El valor por defecto equivale a un borracho en 2 dimensiones
      cuya probabilidad de dar un paso hacia arriba, abajo izquierda
      y derecha es la misma.
    '''
    ########################
    #   MANEJO DE ERROR
    #######################
    for prob in probabilidades :
	#Corrobora si la tupla es de numeros, si no lo es imprime uso
	if not mpl.cbook.is_numlike(prob):
	    print('Las probabilidades deben ser caracteres numericos\n')
            uso()
	    sys.exit(0)

    if np.sum(probabilidades) != 1:
	print('Suma de probabilidades diferente de cero\n')
	uso()
	sys.exit(0)

    #################################
    #   CALCULO DE FRECUENCIAS
    ################################

    if len(probabilidades) == 2 :
        #Borracho en una dimension
	muestras=[] #el muestreo total despues de poner al borracho a caminar N veces
	for sim in xrange(0,N): # corre las N simulaciones
	    x=0 #empieza a caminar desde cero
	    pasos=np.random.rand(n) #los pasos del borracho
	    for p in pasos: #los traducimos al cristiano
	        if p<=probabilidades[0]:  x+=1
		elif p>1-probabilidades[1]: x-=1
	    muestras.append(x) # se recolectan las posiciones finales
	fig=plt.figure()
        m=np.array(muestras)
        mu=np.mean(m)
        sd=np.sqrt(np.mean(m*m)-np.power(np.mean(m),2))
	media, sigma = norm.fit(muestras) #se calculan para dibujar la gaussiana envolvente
	freqs,bins,patches=plt.hist(muestras,bins=70,normed=True,alpha=0.60,facecolor='green') #histograma de frecuencias
	factor=np.max(freqs)*sigma*np.sqrt(2*np.pi) #normaliza la altura al nivel de la frecuencia maxima
	envolvente=factor*mlab.normpdf(bins,media,sigma) #la distribucion normal es una gaussiana
        legend='media%.2f sigma %.2f'%(media,sigma)
	plt.plot(bins,envolvente,'r',label=legend) #se dibuja la distribucion normal
        print(media,sigma)


        plt.xlabel("posicion final")
	plt.ylabel("frecuencia")
	plt.title("Problema de camino aleatorio unidimensional con p = "+repr(probabilidades[0])+" y q = "+repr(probabilidades[1]))
        plt.legend()

	plt.show()
        return mu,sd
	

    elif len(probabilidades) == 4 :
	#borracho en dos dimensiones
	muestras=[]
	muestrasx=[]
	muestrasy=[]
	camino={'x':[0],'y':[0]}
	for sim in xrange(0,N):
	    x=0
	    y=0
	    pasos=np.random.rand(n)
	    for p in pasos:
		if p<=probabilidades[0]: x+=1
		elif p>probabilidades[0] and p<=probabilidades[1]+probabilidades[0]: x-=1
		elif p<=1-probabilidades[3] and p>1-probabilidades[3]-probabilidades[2]: y+=1
		elif p>1-probabilidades[3]: y-=1
		if sim == N-1:
		    camino['x']=np.append(camino['x'],x)
		    camino['y']=np.append(camino['y'],y)
	    muestras.append(np.sqrt(x*x+y*y)) #recolecta la distancia para hacer un hostograma en una dimension
	    muestrasx.append(x)
	    muestrasy.append(y)
	fig=plt.figure()
	plt.title("Probelma del camino aleatorio en 2 dimensiones")

        #histograma de la distancia con su envolvente
	fig.add_subplot(221)
	media,sigma = norm.fit(muestras) #se calculan para dibujar la gaussiana envolvente
        freqs,bins,patches=plt.hist(muestras,bins=70,normed=True,alpha=0.60,facecolor='green') #histograma de frecuencias
        factor=np.max(freqs)*sigma*np.sqrt(2*np.pi) #normaliza la altura al nivel de la frecuencia maxima
        envolvente=factor*mlab.normpdf(bins,media,sigma) #la distribucion normal es una gaussiana
        plt.plot(bins,envolvente,'r') #se dibuja la distribucion normal

	plt.xlabel("Distancia desde la posicion inicial")
	plt.ylabel("Frecuecnia")
	#plt.title("Histograma de la distancia comparada co una Gaussiana")

        #histogramas para x e y independientes con su envolvente
	fig.add_subplot(222)
	media,sigma=norm.fit(muestrasx)
	freqs,bins,patches=plt.hist(muestrasx,bins=70,normed=True,alpha=0.6,facecolor='blue')
	factor=np.max(freqs)*sigma*np.sqrt(2*np.pi)
	envolvente=factor*mlab.normpdf(bins,media,sigma)
	plt.plot(bins,envolvente,'r')
	media,sigma=norm.fit(muestrasy)
	freqs,bins,patches=plt.hist(muestrasy,bins=70,normed=True,alpha=0.6,facecolor='green')
	factor=np.max(freqs)*sigma*np.sqrt(2*np.pi)
	envolvente=factor*mlab.normpdf(bins,media,sigma)
	plt.plot(bins,envolvente,'r')

	plt.xlabel("Posiciones finales de x e y")
	plt.ylabel("Frecuencias")
	#plt.title("Histogramas de x e y y sus envolventes")

        #dibujo del camino
	fig.add_subplot(223)
	plt.plot(camino['x'],camino['y'])

	#plt.xlabel("x")
	#plt.ylabel("y")
	#plt.title("Uno de los N caminos tomados")

        #histograma bidimensional
        fig.add_subplot(224)
	H,bordesx,bordesy = np.histogram2d(muestrasx,muestrasy, bins=(50,50), normed=True)
	tamano=[bordesy[0],bordesy[-1],bordesx[0],bordesx[-1]]
	plt.imshow(H,extent=tamano,interpolation='nearest')
	plt.colorbar()

	#plt.xlabel("x")
	#plt.ylabel("y")
	#plt.title("Histograma bidimensional")

	plt.show()
		

    elif len(probabilidades) == 8 :
        #borracho en dos dimensiones con posibilidad de moverse en las diagonales
	muestras=[]
	muestrasx=[]
	muestrasy=[]
	camino={'x':[0],'y':[0]}
	for sim in xrange(0,N):
	    x=0
	    y=0
	    pasos=np.random.rand(n)
	    for p in pasos:
		if p<=probabilidades[0]: x+=1
		elif p>probabilidades[0] and p<=probabilidades[1]+probabilidades[0]: x-=1
		elif p>probabilidades[1]+probabilidades[0] and p<=probabilidades[1]+probabilidades[0]+probabilidades[2]: y+=1
		elif p>probabilidades[1]+probabilidades[0]+probabilidades[2] and p<=probabilidades[1]+probabilidades[0]+probabilidades[2]+probabilidades[3]: y-=1
                elif p>probabilidades[1]+probabilidades[0]+probabilidades[2]+probabilidades[3] and p<=probabilidades[0]+probabilidades[1]+probabilidades[2]+probabilidades[3]+probabilidades[4]:
		    x+=np.sqrt(2)/2.0
		    y+=np.sqrt(2)/2.0
		elif p>probabilidades[1]+probabilidades[0]+probabilidades[2]+probabilidades[3]+probabilidades[4] and p<=probabilidades[0]+probabilidades[1]+probabilidades[2]+probabilidades[3]+probabilidades[4]+probabilidades[5]:
		    x+=np.sqrt(2)/2.0
		    y-=np.sqrt(2)/2.0
	        elif p>1-probabilidades[7]-probabilidades[6] and p<=1-probabilidades[7]:
		    x-=np.sqrt(2)/2.0
		    y+=np.sqrt(2)/2.0
	        elif p>1-probabilidades[7]:
		    x-=np.sqrt(2)/2.0
		    y-=np.sqrt(2)/2.0
		if sim == N-1:
		    camino['x']=np.append(camino['x'],x)
		    camino['y']=np.append(camino['y'],y)
	    muestras.append(np.sqrt(x*x+y*y)) #recolecta la distancia para hacer un hostograma en una dimension
	    muestrasx.append(x)
	    muestrasy.append(y)
	fig=plt.figure()
	plt.title("Problema del camino aleatorio en 2 dimensiones")

        #histograma de la distancia con su envolvente
	fig.add_subplot(221)
	media,sigma = norm.fit(muestras) #se calculan para dibujar la gaussiana envolvente
        freqs,bins,patches=plt.hist(muestras,bins=70,normed=True,alpha=0.60,facecolor='green') #histograma de frecuencias
        factor=np.max(freqs)*sigma*np.sqrt(2*np.pi) #normaliza la altura al nivel de la frecuencia maxima
        envolvente=factor*mlab.normpdf(bins,media,sigma) #la distribucion normal es una gaussiana
        plt.plot(bins,envolvente,'r') #se dibuja la distribucion normal

	plt.xlabel("Distancia desde la posicion inicial")
	plt.ylabel("Frecuencia")
	#plt.title("Histograma de la distancia comparada con una Gaussiana")

        #histogramas para x e y independientes con su envolvente
	fig.add_subplot(222)
	media,sigma=norm.fit(muestrasx)
	freqs,bins,patches=plt.hist(muestrasx,bins=70,normed=True,alpha=0.6,facecolor='blue')
	factor=np.max(freqs)*sigma*np.sqrt(2*np.pi)
	envolvente=factor*mlab.normpdf(bins,media,sigma)
	plt.plot(bins,envolvente,'r')
	media,sigma=norm.fit(muestrasy)
	freqs,bins,patches=plt.hist(muestrasy,bins=70,normed=True,alpha=0.6,facecolor='green')
	factor=np.max(freqs)*sigma*np.sqrt(2*np.pi)
	envolvente=factor*mlab.normpdf(bins,media,sigma)
	plt.plot(bins,envolvente,'r')

	plt.xlabel("x e y")
	plt.ylabel("Frecuencias")
	#plt.title("Histogramas para x e y")

        #dibujo del camino
	fig.add_subplot(223)
	plt.plot(camino['x'],camino['y'])

	#plt.xlabel("x")
	#plt.ylabel("y")
	#plt.title("Uno de los N caminos tomados")

        #histograma bidimensional
        fig.add_subplot(224)
	H,bordesx,bordesy = np.histogram2d(muestrasx,muestrasy, bins=(50,50), normed=True)
	tamano=[bordesy[0],bordesy[-1],bordesx[0],bordesx[-1]]
	plt.imshow(H,extent=tamano,interpolation='nearest')
	plt.colorbar()

	#plt.xlabel("x")
	#plt.ylabel("y")
	#plt.title("Histograma bidimensional")

	plt.show()


    elif len(probabilidades):
	#El programa no esta disegnado para calcular otras dimensiones, se imprime uso
	print('la longitud de la tupla debe ser de 2, 4 u 8, vea ?drunkard\n')
	uso()
	sys.exit(0)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    Problema de 2 borrachos 1d
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def drunkar2(n,N,prob1=(0.5,0.5),prob2=(0.5,0.5)):
    ''' Muestra la probabilidad de que 2 caminos aleatorios se crucen
    dependiendo del numero n de pasos

    Parameters
    ----------
    n : entero
        Numero maximo de pasos, comienza en 50.
    N : entero
        Numero de veces que se pone a caminar los borrachos.
    prob1,2 : tupla
        Tupla que determina las probabilidades de ir en cada direccion
        Cada valor debe estar entre 0 y 1.
        Los valores por defecto son de igual probabilidad para ambos borrachos.
    '''
    ########################
    #   MANEJO DE ERROR
    #######################
    for p1,p2 in prob1,prob2 :
    #Corrobora si la tupla es de numeros, si no lo es imprime uso
        if not mpl.cbook.is_numlike(p1) or not mpl.cbook.is_numlike(p2):
            print('Las probabilidades deben ser caracteres numericos\n')
            uso2()
            sys.exit(0)
    if np.sum(prob1) != 1 or np.sum(prob2) != 1:
        print('Suma de probabilidades diferente de cero\n')
        uso2()
        sys.exit(0)

    muestras=np.zeros(n-50) #lista de las veces que se encuentran los borrachos
    for pasos in xrange(50,n):
	for sim in xrange(0,N):
	    x=0 #empieza a caminar desde cero
	    y=0
	    pasosx=np.random.rand(pasos) #los pasos del borracho
	    pasosy=np.random.rand(pasos)
	    for px in pasosx: #los traducimos al cristiano
	        if px<=prob1[0]:  x+=1
		elif px>1-prob1[1]: x-=1
	    for py in pasosy: #los traducimos al cristiano
	        if px<=prob1[0]:  x+=1
		if py<=prob2[0]: y+=1
		elif py<1-prob2[1]: y-=1
	    if x==y: muestras[pasos-50]+=1 #si los borrachos se encuentran
    #print(muestras) 
    fig=plt.figure()
    #media, sigma = norm.fit(muestras) #se calculan para dibujar la gaussiana envolvente
    #freqs,bins,patches=plt.hist(muestras,bins=70,normed=True,alpha=0.60,facecolor='green') #histograma de frecuencias
    plt.plot(muestras,'o')
    plt.xlabel("Conteo de encuentros")
    plt.ylabel("Frecuencia")
    plt.title("Problema de los 2 borrachos")

    plt.show()

   

if __name__== "__main__":

    #implementacion del borracho en 1 dimension
    drunkard(300,10000,(0.5,0.5)) #iguales probabilidades
    mu,sd=drunkard(300,10000,(0.4,0.6)) #probabilidades diferentes

    sd_list=[]
    N_list=[50,100,150,250,500,700,800,1000]
    for N in N_list:
       mu,sd=drunkard(N,10000,(0.5,0.5))
       sd_list.append(sd)
    plt.figure()
    N=np.linspace(50,1000,100)
    plt.plot(N_list,np.power(sd_list,2),'ro',label="Simulacion")
    plt.plot(N,N,label="Linea de pendiente 1")
    plt.legend()
    plt.ylabel("Deviacion estandar")
    plt.xlabel("N")
    plt.show()
   

   # #implementacion del borracho en 2 dimensiones con 4 posibilidades de movimiento
   # drunkard(200,1000,(0.05,0.45,0.35,0.15)) #probabilidades diferentes
   # drunkard(200,1000,(0.25,0.25,0.25,0.25)) #probabilidades iguales

    #implementacion del borracho en 2 dimensiones con 8 posibilidades de movimiento
    #probs=tuple(np.linspace(0.125,0.125,8))
    #drunkard(300,1000,probs) #probanilidades iguales
    #pass
