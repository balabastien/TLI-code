import sys, os
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm




#### CODE FOR COMPUTING SURFACE CHERN NUMBER IN Z DIRECTION #######







##### GIVE SYSTEM SIZE #######
Nx = 15
Ny = 15
Nz = 11

cutoff = np.floor(Nz/2)-2

##### VARIABLE INITIALIZATION #######
latticeleft1=np.zeros(2*Nx*Ny*Nz)
latticeleft2=np.zeros(2*Nx*Ny*Nz)
vectorpr1=[]
vectorpr2=[]
vectorpr1normed=[]
vectorpr2normed=[]
projector2=np.zeros((2*Nx*Ny,2*Nx*Ny),dtype=np.complex128)
projector1= np.zeros((2*Nx*Ny,2*Nx*Ny),dtype=np.complex128)
hamiltonian= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)

latticeleft=np.zeros(2*Nx*Ny*Nz)
operatorx = np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
operatory= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
operatorz= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))





### LATTICE VECTORS #####
def latticematrix2d(x,y,m):
    latticematrix = np.zeros((Nx,Ny,2))
    latticematrix[x][y][m] = 1
    return latticematrix

def latticevector2d(x,y,m):
    latticevector = np.array(latticematrix2d(x%Nx ,y%Ny ,m))
    latticevectornew = latticevector.reshape(2*Ny*Nx)
    return  latticevectornew

def latticevector3d(x,y,z,m):
   liste = np.zeros(2*Nz*Ny*Nx)
   nonzeroposition = np.nonzero(latticevector2d(x,y,m))[0]
   liste[(z*2*Nx*Ny+nonzeroposition)]=1
   return liste



#### Bloch eigenvectors of Chern insulator #####

def vectorpr1(kx,ky):
    return [(np.sqrt(3+2*np.cos(kx)*(np.cos(ky)-1)-2*np.cos(ky))-np.sin(ky)),-1+np.cos(kx)+np.cos(ky)-1j*np.sin(kx)]

def vectorpr2(kx,ky):
    return [-(np.sqrt(3+2*np.cos(kx)*(np.cos(ky)-1)-2*np.cos(ky))+np.sin(ky)),-1+np.cos(kx)+np.cos(ky)-1j*np.sin(kx)]

def vectorpr1normed(kx,ky):
    return vectorpr1(kx,ky)*1/(LA.norm(vectorpr1(kx,ky)))
def vectorpr2normed(kx,ky):
    return vectorpr2(kx,ky)*1/(LA.norm(vectorpr2(kx,ky)))


### Extension to the 2D lattice

def extension1(i, j, l, kx, ky):
    return 1/(np.sqrt(Nx*Ny))*np.exp(-1j*(kx*i+ky*j))*vectorpr1normed(kx,ky)[l]
def extension2(i, j, l, kx, ky):
    return 1/(np.sqrt(Nx*Ny))*np.exp(-1j*(kx*i+ky*j))*vectorpr2normed(kx,ky)[l]

def vector1(kx,ky):
    return [[[[extension1(i, j, l, kx, ky) for l in range(2)] for j in range(Ny)] for i in range(Nx)]]
def vector2(kx,ky):
    return [[[[extension2(i, j, l, kx, ky) for l in range(2)] for j in range(Ny)] for i in range(Nx)]]

def vector1reshape(kx,ky):
    vectorfirst = np.array(vector1(kx,ky))
    vector1new = vectorfirst.reshape(2*Ny*Nx)
    return vector1new

def vector2reshape(kx,ky):
    vectorfirst = np.array(vector2(kx,ky))
    vector1new = vectorfirst.reshape(2*Ny*Nx)
    return vector1new



### Projectors


for X in range(1,Nx+1):
    for Y in range(1,Ny+1):
       projector1+=np.outer(np.conj(vector1reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny)),vector1reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny))
       projector2+=np.outer(np.conj(vector2reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny)),vector2reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny))

### Extension to the 3D lattice

def vector1reshapefull(kx,ky,z):
    vector1 = np.zeros((2*Nx*Ny*Nz),dtype=np.complex128)
    vector1[z*2*Nx*Ny:2*Nx*Ny*(z+1)] = vector1reshape(kx,ky)
    return vector1


def vector2reshapefull(kx,ky,z):
    vector2 = np.zeros((2*Nx*Ny*Nz),dtype=np.complex128)
    vector2[z*2*Nx*Ny:2*Nx*Ny*(z+1)] = vector2reshape(kx,ky)
    return vector2


### Momentum dependent projector
def projectormomentum(kx,ky,z):
    projectormomentum = np.outer(np.conj(vector1reshapefull(kx,ky,z)),vector1reshapefull(kx,ky,z))+np.outer(np.conj(vector2reshapefull(kx,ky,z+1)),vector2reshapefull(kx,ky,z+1))
    return projectormomentum



def trialwavefunction(x,y,z,m):
    return (latticevector3d(x,y,z,m) +  latticevector3d(x,y,(z+1)%Nz,m))/(np.sqrt(2))






def newwavefunc1(kx,ky,x,y,z):
    return (projectormomentum(kx,ky,z) @ trialwavefunction(x,y,z,1))


def newwavefunc2(kx,ky,x,y,z):
    return projectormomentum(kx,ky,z) @ trialwavefunction(x,y,z,0)






def wannierfunc1(x,y,z,Rx,Ry):
    wannierfunc1 = np.zeros((2*Nx*Ny*Nz),dtype=np.complex128)
    for kx in range(1,Nx+1):
        for ky in range(1,Ny+1):
             wannierfunc1+=np.exp(1j*Rx*2*np.pi*kx/Nx)*np.exp(1j*Ry*2*np.pi*ky/Ny)*newwavefunc1(2*np.pi*kx/Nx,2*np.pi*ky/Ny,x,y,z)
    return wannierfunc1

def wannierfunc2(x,y,z,Rx,Ry):
    wannierfunc2 = np.zeros((2*Nx*Ny*Nz),dtype=np.complex128)
    for kx in range(1,Nx+1):
        for ky in range(1,Ny+1):
             wannierfunc2+=np.exp(1j*Rx*2*np.pi*kx/Nx)*np.exp(1j*Ry*2*np.pi*ky/Ny)*newwavefunc2(2*np.pi*kx/Nx,2*np.pi*ky/Ny,x,y,z)
    return wannierfunc2





wanniercenter1 = wannierfunc1(0,0,0,0,0)
wanniercenter2 = wannierfunc2(0,0,0,0,0)


wanniercenter1tensor = wanniercenter1.reshape(Nz,Nx,Ny,2, order='C')
wanniercenter2tensor = wanniercenter2.reshape(Nz,Nx,Ny,2, order='C')


### Wannier function at all positions obtained by translations


def wannierxyz1(Rx, Ry,Rz):
   return np.roll(wanniercenter1tensor, [Rx,Ry,Rz], axis=(1, 2, 0))



def wannierxyz2(Rx, Ry,Rz):
   return np.roll(wanniercenter2tensor, [Rx,Ry,Rz], axis=(1, 2, 0))



### Open boundary conditions in Z by truncating Wannier functions


def wannierxyz1ntrunc(Rx,Ry,Rz):
    wannierxyz1ntrunc = np.array(wannierxyz1(Rx,Ry,Rz))
    if Rz>int(np.floor(Nz/2)):
        wannierxyz1ntrunc[:(Rz-int(np.floor(Nz/2))),:Nx,:Ny,:2] =0
    if Rz == int(np.floor(Nz/2)):  
        wannierxyz1ntrunc = wannierxyz1(int(np.floor(Nz/2)),Rx,Ry)
    if Rz < int(np.floor(Nz/2)):                                   
        wannierxyz1ntrunc[int(np.floor(Nz/2))+Rz:Nz,:Nx,:Ny,:2] =0
    return wannierxyz1ntrunc


def wannierxyz2ntrunc(Rx,Ry,Rz):
    wannierxyz2ntrunc = np.array(wannierxyz2(Rx,Ry,Rz))
    if Rz>int(np.floor(Nz/2)):
        wannierxyz2ntrunc[:(Rz-int(np.floor(Nz/2))),:Nx,:Ny,:2] =0
    if Rz == int(np.floor(Nz/2)):  
        wannierxyz2ntrunc = wannierxyz2(int(np.floor(Nz/2)),Rx,Ry)
    if Rz < int(np.floor(Nz/2)):                                   
        wannierxyz2ntrunc[int(np.floor(Nz/2))+Rz:Nz,:Nx,:Ny,:2] =0
    return wannierxyz2ntrunc




def wanniersingle(x,y,z,m):
    if m == 0:
       wanniersingle1 = np.array(wannierxyz1ntrunc(x, y,z))
       wanniersingle1new = wanniersingle1.reshape(2*Nz*Ny*Nx)
       return wanniersingle1new * np.sqrt(2)
    if m == 1:
       wanniersingle2 = np.array(wannierxyz2ntrunc(x, y,z))
       wanniersingle2new = wanniersingle2.reshape(2*Nz*Ny*Nx)
       return wanniersingle2new * np.sqrt(2)


### TLI Hamiltonian
for x in range(Nx):
    for y in range(Ny):
        for z in range(Nz):
            for m in range(2):
                hamiltonian += np.random.uniform(-1,1)*np.outer(np.conj(wanniersingle(x,y,z,m)),wanniersingle(x,y,z,m))



### All hoppings terms we add
hoppingterm= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for z in range(Nz-1):
    for x in range(Nx):
        for y in range(Ny):
            for m in range(2):
                   hoppingterm += (np.outer(latticevector3d(x,y,z,m),latticevector3d(x+1,y,z,m))+ np.outer(latticevector3d(x,y,z,m),latticevector3d(x,y,z+1,m))+ np.outer(latticevector3d(x,y,z,m),latticevector3d(x,y+1,z,m)))
   

hoppingterminterband= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for z in range(Nz-1):
    for x in range(Nx):
        for y in range(Ny):
                   hoppingterminterband += (np.outer(latticevector3d(x,y,z,0),latticevector3d(x+1,y,z,1))+ np.outer(latticevector3d(x,y,z,0),latticevector3d(x,y,z+1,1))+ np.outer(latticevector3d(x,y,z,0),latticevector3d(x,y+1,z,1))) + (np.outer(latticevector3d(x,y,z,1),latticevector3d(x+1,y,z,0))+ np.outer(latticevector3d(x,y,z,1),latticevector3d(x,y,z+1,0))+ np.outer(latticevector3d(x,y,z,1),latticevector3d(x,y+1,z,0)))
   



boundarytermhopping= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for x in range(Nx):
    for y in range(Ny):
        for m in range(2):
             boundarytermhopping += (np.outer(latticevector3d(x,y,Nz-1,m),latticevector3d(x+1,y,Nz-1,m))+ np.outer(latticevector3d(x,y,Nz-1,m),latticevector3d(x,y+1,Nz-1,m))) 
   

boundarytermhoppinginterband= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for x in range(Nx):
    for y in range(Ny):
             boundarytermhoppinginterband += (np.outer(latticevector3d(x,y,Nz-1,0),latticevector3d(x+1,y,Nz-1,1))+ np.outer(latticevector3d(x,y,Nz-1,0),latticevector3d(x,y+1,Nz-1,1)))  + (np.outer(latticevector3d(x,y,Nz-1,1),latticevector3d(x+1,y,Nz-1,0))+ np.outer(latticevector3d(x,y,Nz-1,1),latticevector3d(x,y+1,Nz-1,0))) 
   




### Numerical coef for Prodan Chern number formula c_m

def clambda(l,N):
    return np.exp(2*np.pi*1j*l/N * (np.floor(N/2)+1))/(1- np.exp(2*np.pi*l*1j/N))






for i in range(Nx):
    for j in range(Ny):
        for k in range(int(np.floor(Nz/2))-int(cutoff)):
            for m in range(2):
                latticeleft1 += latticevector3d(i,j,k,m)



for i in range(Nx):
    for j in range(Ny):
        for k in range(int(np.floor(Nz/2))+int(cutoff),Nz):
            for m in range(2):
                latticeleft2 += latticevector3d(i,j,k,m)


latticeleft = latticeleft1+latticeleft2

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            for m in range(2):
                operatorx = operatorx + (i+1)*np.outer(np.conj(latticevector3d(i,j,k,m)),latticevector3d(i,j,k,m))
                operatory = operatory +(j+1)*np.outer(np.conj(latticevector3d(i,j,k,m)),latticevector3d(i,j,k,m))
                operatorz = operatorz + (k+1)*np.outer(np.conj(latticevector3d(i,j,k,m)),latticevector3d(i,j,k,m))



def matrixoperatorx(m):
    return np.diag(np.exp(1j*2*np.pi*m/(Nx)*np.diag(operatorx)))

def matrixoperatory(m):
    return np.diag(np.exp(1j*2*np.pi*m/(Ny)*np.diag(operatory)))

def matrixoperatorz(m):
    return np.diag(np.exp(1j*2*np.pi*m/(Nz)*np.diag(operatorz)))
 
def arraytoindex(i,j,k,m):
    return k*2*Nx*Ny+2*j+2*Ny*i+m

leftsystem = []
for i in range(Nx):
    for j in range(Ny):
        for k in range(int(np.floor(Nz/2))):
            for m in range(2):
                leftsystem.append(arraytoindex(i,j,k,m))

def submatrixoperatorx(l):
    return matrixoperatorx(l)[leftsystem,:][:,leftsystem]
def submatrixoperatory(l):
    return matrixoperatory(l)[leftsystem,:][:,leftsystem]
     



numbereigenvectors=[]
chernlist = []



#### COMPUTE CHERN NUMBER FOR DIFFERENT HOPPING STRENGTHS
tspace = np.linspace(0,0.11,50)
for t in tspace:
   hamiltoniannew = hamiltonian +t*(hoppingterm + hoppingterm.conj().T + boundarytermhopping + boundarytermhopping.conj().T + (hoppingterminterband + hoppingterminterband.conj().T + boundarytermhoppinginterband + boundarytermhoppinginterband.conj().T)/10) 
   projector = np.identity(2*Nx*Ny*Nz,dtype=np.complex128)
   commuty = np.zeros((len(leftsystem),len(leftsystem)),dtype=np.complex128)
   commutx = np.zeros((len(leftsystem),len(leftsystem)),dtype=np.complex128)
   superproj=[]
   chernprodan=0
   w, v = LA.eigh(hamiltoniannew)    
   vectormask = (np.linalg.norm(latticeleft*np.transpose(v), axis=1) < 0.1).astype(bool)
   listevectors=[]
   listevectors = np.transpose(v)[vectormask,:]
   numbereigenvectors.append(len(listevectors))
   print(len(listevectors))
   for i in range(0,len(listevectors)):
       projector -= np.outer(np.conj(listevectors[i]),listevectors[i])
    
   subprojector = projector[leftsystem,:][:,leftsystem]


     
    
   for l in range(1,Nx):
       commuty += clambda(l,Ny)* ( submatrixoperatory(l) @ subprojector @ submatrixoperatory(-l))
       commutx += clambda(l,Nx)* ( submatrixoperatorx(l) @ subprojector @ submatrixoperatorx(-l)) 
   commutator = np.dot(commuty,commutx)-np.dot(commutx,commuty)
   superproj = np.dot(subprojector,commutator)
   chernprodan = -2*np.pi*1j/(Ny*Nx) * np.trace(superproj)
   print(chernprodan)
   chernlist.append(chernprodan)


plt.plot(tspace,chernlist)
plt.show()


a_file = open("chern151511.txt", "w")
np.savetxt(a_file, chernlist)
a_file.close()







