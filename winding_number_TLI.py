import sys, os
from numpy import linalg as LA
import numpy as np
import pprint
import matplotlib.pyplot as plt
from scipy.linalg import expm
from collections import Counter




Nx = 11
Ny = 11
Nz = 11

vectorpr1=[]
vectorpr2=[]
vectorpr1normed=[]
vectorpr2normed=[]
projector2=np.zeros((2*Nx*Ny,2*Nx*Ny),dtype=np.complex128)
projector1= np.zeros((2*Nx*Ny,2*Nx*Ny),dtype=np.complex128)
hamiltonian= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)
hamiltoniannew= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)
pbcterm= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)

latticeleft=np.zeros(2*Nx*Ny*Nz)
operatorx = np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
operatory= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
operatorz= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))







def convert_angle_to_0_2pi_interval(angle):
    new_angle = np.arctan2(np.sin(angle), np.cos(angle))
    if new_angle < 0:
        new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
    return new_angle


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

### Here we define the lattice vectors.


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



#### Bloch eigenvectors

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


for X in range(1,Nx+1):
    for Y in range(1,Ny+1):
       projector1+=np.outer(np.conj(vector1reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny)),vector1reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny))
       projector2+=np.outer(np.conj(vector2reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny)),vector2reshape(2*np.pi*X/Nx,2*np.pi*Y/Ny))



def vector1reshapefull(kx,ky,z):
    vector1 = np.zeros((2*Nx*Ny*Nz),dtype=np.complex128)
    vector1[z*2*Nx*Ny:2*Nx*Ny*(z+1)] = vector1reshape(kx,ky)
    return vector1


def vector2reshapefull(kx,ky,z):
    vector2 = np.zeros((2*Nx*Ny*Nz),dtype=np.complex128)
    vector2[z*2*Nx*Ny:2*Nx*Ny*(z+1)] = vector2reshape(kx,ky)
    return vector2




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



wanniercenter1 = wannierfunc1(0,0,0,0,0)
wanniercenter2 = wannierfunc2(0,0,0,0,0)


wanniercenter1tensor = wanniercenter1.reshape(Nz,Nx,Ny,2, order='C')
wanniercenter2tensor = wanniercenter2.reshape(Nz,Nx,Ny,2, order='C')



def wannierxyz1(Rx, Ry,Rz):
   return np.roll(wanniercenter1tensor, [Rx,Ry,Rz], axis=(1, 2, 0))



def wannierxyz2(Rx, Ry,Rz):
   return np.roll(wanniercenter2tensor, [Rx,Ry,Rz], axis=(1, 2, 0))





def wanniersingle(x,y,z,m):
    if m == 0:
       wanniersingle1 = np.array(wannierxyz1(x, y,z))
       wanniersingle1new = wanniersingle1.reshape(2*Nz*Ny*Nx)
       return wanniersingle1new * np.sqrt(2)
    if m == 1:
       wanniersingle2 = np.array(wannierxyz2(x, y,z))
       wanniersingle2new = wanniersingle2.reshape(2*Nz*Ny*Nx)
       return wanniersingle2new * np.sqrt(2)



for x in range(Nx):
    for y in range(Ny):
        for z in range(Nz):
            for m in range(2):
                hamiltonian += np.random.uniform(-1,1)*np.outer(np.conj(wanniersingle(x,y,z,m)),wanniersingle(x,y,z,m))




hoppingterm= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for z in range(Nz-1):
    for x in range(Nx):
        for y in range(Ny):
            for m in range(2):
                   hoppingterm += (np.outer(latticevector3d(x,y,z,m),latticevector3d(x+1,y,z,m))+np.outer(latticevector3d(x,y,z,m),latticevector3d(x,y+1,z,m)) + np.outer(latticevector3d(x,y,z,m),latticevector3d(x,y,z+1,m)))
   
     
hoppingterminterband= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for z in range(Nz-1):
    for x in range(Nx):
        for y in range(Ny):
                   hoppingterminterband += (np.outer(latticevector3d(x,y,z,0),latticevector3d(x+1,y,z,1))+ np.outer(latticevector3d(x,y,z,0),latticevector3d(x,y,z+1,1))+ np.outer(latticevector3d(x,y,z,0),latticevector3d(x,y+1,z,1))) + (np.outer(latticevector3d(x,y,z,1),latticevector3d(x+1,y,z,0))+ np.outer(latticevector3d(x,y,z,1),latticevector3d(x,y,z+1,0))+ np.outer(latticevector3d(x,y,z,1),latticevector3d(x,y+1,z,0)))
   



hoppingpbc = np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for y in range(Ny):
    for x in range(Nx):
        for m in range(2):
            hoppingpbc += np.outer(latticevector3d(x,y,Nz-1,m),latticevector3d(x,y,0,m)) + np.outer(latticevector3d(x,y,Nz-1,m),latticevector3d(x+1,y,Nz-1,m)) +  + np.outer(latticevector3d(x,y,Nz-1,m),latticevector3d(x,y+1,Nz-1,m))




boundarytermhoppinginterbandpbc= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for x in range(Nx):
    for y in range(Ny):
             boundarytermhoppinginterbandpbc += (np.outer(latticevector3d(x,y,Nz-1,0),latticevector3d(x+1,y,Nz-1,1))+ np.outer(latticevector3d(x,y,Nz-1,0),latticevector3d(x,y+1,Nz-1,1)))  + np.outer(latticevector3d(x,y,Nz-1,0),latticevector3d(x,y+1,0,1))   + (np.outer(latticevector3d(x,y,Nz-1,1),latticevector3d(x+1,y,Nz-1,0))+ np.outer(latticevector3d(x,y,Nz-1,1),latticevector3d(x,y+1,Nz-1,0))) + np.outer(latticevector3d(x,y,Nz-1,1),latticevector3d(x,y+1,0,0)) 
   






def clambda(l,N):
    return np.exp(2*np.pi*1j*l/N * (np.floor(N/2)+1))/(1- np.exp(2*np.pi*l*1j/N))
def arraytoindex(i,j,k):
    return k*2*Nx*Ny+2*j+2*Ny*i
arraylist=[]
arraylist2=[]
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            arraylist.append(arraytoindex(i,j,k))
            arraylist2.append(arraytoindex(i,j,k)+1)
   
ordering = np.argsort(np.insert(arraylist2, np.arange(len(arraylist)), arraylist))




## Make the loop running over hopping t_1 here!

interval=np.linspace(0,0.11,50)
windinglist=[]
for t1 in interval:
   print(t1)
   hamiltoniannew = hamiltonian + t1*(hoppingterm + hoppingterm.conj().T + hoppingpbc + hoppingpbc.conj().T + (hoppingterminterband + hoppingterminterband.conj().T + boundarytermhoppinginterbandpbc + boundarytermhoppinginterbandpbc.conj().T)/10)
   w, v = LA.eigh(hamiltoniannew)    


   def angloz(i):
        return ((convert_angle_to_0_2pi_interval(np.angle(np.dot(np.conj(v[:,i]),matrixoperatorz(1)@ v[:,i] )))/(2*np.pi/Nz))+0.5)%Nz


   def anglez(i,n):
           return min(n-i, Nz-(n-i))


   def anglox(i):
        return ((convert_angle_to_0_2pi_interval(np.angle(np.dot(np.conj(v[:,i]),matrixoperatorx(1)@ v[:,i] )))/(2*np.pi/Nx)))%Nx


   def anglex(i,n):
           return min(n-i, Nx-(n-i))



   def angloy(i):
        return ((convert_angle_to_0_2pi_interval(np.angle(np.dot(np.conj(v[:,i]),matrixoperatory(1)@ v[:,i] )))/(2*np.pi/Ny)))%Ny

   def angley(i,n):
           return min(n-i, Ny-(n-i))


   def distancetorus(i,j,k,n1,n2,n3):
       return np.sqrt(anglex(n1,i)**2+angley(n2,j)**2+anglez(n3,k)**2)



   angletablex=[]
   angletabley=[]
   angletablez=[]
   for i in range(len(w)):
       angletablex.append(anglox(i))
       angletabley.append(angloy(i))
       angletablez.append(angloz(i))


   arraydistance=[]
   for m in range(len(w)):
      for i in range(1,Nx+1):
          for j in range(1,Ny+1):
              for k in range(1,Nz+1):
                   arraydistance.append(distancetorus(j,k,i,angletablex[m],angletabley[m],angletablez[m]))
   distancetensor=np.array(arraydistance).reshape(len(w),Nz*Ny*Nx)
   distancetensorsorted = np.argsort(distancetensor, axis=0)
   print("this tensor is constructed, damn!")
   Ax = np.zeros(Nx*Ny*Nz)
   Bx = []
   while(len(Bx)<2*Nx*Ny*Nz):
       for i,k in enumerate(Ax):
           while(k<Nx*Ny*Nz):
               if Bx.count(int(distancetensorsorted[int(k),i]))==1:
                   k+=1
               else:
                   Ax[i]=k+1
                   Bx.append(distancetensorsorted[int(k),i])
                   break          
   correct_order = np.insert(Bx[len(Bx)//2:], np.arange(len(Bx[:len(Bx)//2])),Bx[:len(Bx)//2])   
       
   sorted_v = v[:,correct_order]
   bracketx= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)
   brackety= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)
   bracketz= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz),dtype=np.complex128)
   for l in range(1,Nx):
       bracketx += clambda(l,Nx)* ( matrixoperatorx(l) @ sorted_v @ matrixoperatorx(-l))
       brackety += clambda(l,Ny)* ( matrixoperatory(l) @ sorted_v @ matrixoperatory(-l))
       bracketz += clambda(l,Nz)* ( matrixoperatorz(l) @ sorted_v @ matrixoperatorz(-l))

   inversev = LA.inv(sorted_v)
   Wx =  inversev @ bracketx
   Wy =  inversev @ brackety
   Wz =  inversev @ bracketz
   winding = 1j*np.pi/3*(np.trace(Wx@Wy@Wz) - np.trace(Wy@Wx@Wz) - np.trace(Wx@Wz@Wy) - np.trace(Wz@Wy@Wx) + np.trace(Wz@Wx@Wy) + np.trace(Wy@Wz@Wx) )/(Nx*Ny*Nz)
   print(winding)
   windinglist.append(winding)
   
plt.plot(interval,windinglist)
plt.show()

a_file = open("winding111111.txt", "w")
np.savetxt(a_file, windinglist)
a_file.close()









