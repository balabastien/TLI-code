import sys, os
from numpy import linalg as LA
import numpy as np
import pprint
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, eigsh
from scipy import sparse
from scipy.sparse import save_npz, load_npz


def save_sparse_lil(filename, array):
    # use np.savez_compressed(..) for compression
    np.savez(filename, dtype=array.dtype.str, data=array.data,
        rows=array.rows, shape=array.shape)

def load_sparse_lil(filename):
    loader = np.load(filename, allow_pickle=True)
    result = sparse.lil_matrix(tuple(loader["shape"]), dtype=str(loader["dtype"]))
    result.data = loader["data"]
    result.rows = loader["rows"]
    return result


unitary=load_npz('Umatrixsparse131313.npz')


disorderrealisation = 1000
Nx = 13
Ny = 13
Nz = 13
vectorpr1=[]
vectorpr2=[]
vectorpr1normed=[]
vectorpr2normed=[]
wannier0001=[]
wannier0002=[]
matrixsystem= []
linearsolution=[]
latticeleft=np.zeros(2*Nx*Ny*Nz)
operatorx = np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
operatory= np.zeros((2*Nx*Ny*Nz,2*Nx*Ny*Nz))





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



def latticematrix(x,y,z,m):
    latticematrix = np.zeros((Nx,Ny,Nz,2))
    latticematrix[x][y][z][m] = 1
    return latticematrix


def latticevector(x,y,z,m):
    latticevector = np.array(latticematrix(x % Nx,y  % Ny,z  % Nz,m))
    latticevectornew = latticevector.reshape(2*Nz*Ny*Nx)
    return  latticevectornew


matrix2 = sparse.lil_matrix((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
matrix = sparse.lil_matrix((2*Nx*Ny*Nz,2*Nx*Ny*Nz))
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz-1):
             matrix[np.nonzero(latticevector3d(i,j,k,0))[0],np.nonzero(latticevector3d(i+1,j,k,0))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,k,0))[0],np.nonzero(latticevector3d(i,j+1,k,0))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,k,0))[0],np.nonzero(latticevector3d(i,j,k+1,0))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,k,1))[0],np.nonzero(latticevector3d(i+1,j,k,1))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,k,1))[0],np.nonzero(latticevector3d(i,j+1,k,1))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,k,1))[0],np.nonzero(latticevector3d(i,j,k+1,1))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,k,0))[0],np.nonzero(latticevector3d(i+1,j,k,1))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,k,0))[0],np.nonzero(latticevector3d(i,j+1,k,1))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,k,0))[0],np.nonzero(latticevector3d(i,j,k+1,1))[0]] = 1             
             matrix2[np.nonzero(latticevector3d(i,j,k,1))[0],np.nonzero(latticevector3d(i+1,j,k,0))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,k,1))[0],np.nonzero(latticevector3d(i,j+1,k,0))[0]] = 1             
             matrix2[np.nonzero(latticevector3d(i,j,k,1))[0],np.nonzero(latticevector3d(i,j,k+1,0))[0]] = 1             
             

for i in range(Nx):
    for j in range(Ny):
             matrix[np.nonzero(latticevector3d(i,j,Nz-1,0))[0],np.nonzero(latticevector3d(i+1,j,Nz-1,0))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,Nz-1,0))[0],np.nonzero(latticevector3d(i,j+1,Nz-1,0))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,Nz-1,0))[0],np.nonzero(latticevector3d(i,j,0,0))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,Nz-1,1))[0],np.nonzero(latticevector3d(i+1,j,Nz-1,1))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,Nz-1,1))[0],np.nonzero(latticevector3d(i,j+1,Nz-1,1))[0]] = 1
             matrix[np.nonzero(latticevector3d(i,j,Nz-1,1))[0],np.nonzero(latticevector3d(i,j,0,1))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,Nz-1,0))[0],np.nonzero(latticevector3d(i+1,j,Nz-1,1))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,Nz-1,0))[0],np.nonzero(latticevector3d(i,j+1,Nz-1,1))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,Nz-1,0))[0],np.nonzero(latticevector3d(i,j,0,1))[0]] = 1             
             matrix2[np.nonzero(latticevector3d(i,j,Nz-1,1))[0],np.nonzero(latticevector3d(i+1,j,Nz-1,0))[0]] = 1
             matrix2[np.nonzero(latticevector3d(i,j,Nz-1,1))[0],np.nonzero(latticevector3d(i,j+1,Nz-1,0))[0]] = 1             
             matrix2[np.nonzero(latticevector3d(i,j,Nz-1,1))[0],np.nonzero(latticevector3d(i,j,0,0))[0]] = 1      





hoppingmatrix=-(unitary.conj().T)@(matrix + matrix.conj().T)@(unitary)
hoppingmatrix2=-(unitary.conj().T)@(matrix2 + matrix2.conj().T)@(unitary)
#hoppingmatrix=-(matrix + matrix.conj().T)
ratiot1 = []
for W in np.linspace(15,35,20):
   ratio_aveave = []
   for j in range(disorderrealisation):
      a = np.array([np.random.uniform(-1,1) for i in range(2*Nx*Ny*Nz)])
      d = np.diag(a)  
      dnew = sparse.spdiags(a, 0, a.size, a.size)
      matrixnew =   (hoppingmatrix) + hoppingmatrix2*1/10  +W/2*d 
      vals = np.linalg.eigvalsh(matrixnew)
#### Level Spacing statistics ####
      vals.sort()
      spacings = np.diff(vals[int(len(vals) / 2)- 250 : int( len(vals) / 2) + 250])
      spacingsnew = np.delete(spacings,len(spacings)-1)
      spacingsshifted = np.roll(spacings,-1)
      spacingsshiftednew = np.delete(spacingsshifted,len(spacingsshifted)-1)
      ratio_list = np.minimum(spacingsnew,spacingsshiftednew)/(np.maximum(spacingsnew,spacingsshiftednew))
      ratio_ave = np.average(ratio_list)
      ratio_aveave.append(ratio_ave)
   ratiofinal = np.average(ratio_aveave)
   ratiot1.append(ratiofinal)
   print(ratiofinal)
   
a_file = open("levelstat131313.txt", "w")
np.savetxt(a_file, ratiot1)
a_file.close()
   
   

#mumu999=np.array([0.5728477944491948,0.5646141799895072,0.5570888484595475,0.5468459720518812,0.5439070695707994,0.5334269486057849,0.5251867945638353,0.5182382950839585,0.5045760220269547,0.5000400841046776,0.4916038950195519,0.4822315570899514,0.4779333738104499,0.468866438354088,0.46430706930023036,0.4553164589543903,0.453543308168398,0.4502988722977657,0.4458387800606885,0.4407140800497063])
#mumu131313=np.array([0.590385308644351,0.5903624275203609,0.5861521792186825,0.5695413608312732,0.5527948265835428,0.5316987553978241,0.5312134916604705,0.5119512951559819,0.49568045383334514,0.4852296110540958,0.48609281054627845,0.4646053631615913,0.46566970146854086,0.4489030478122273,0.4446470445558955,0.44078609200191965,0.43493857013130055,0.4289647264001849,0.4291861356181716,0.42523108345769345])
#plt.plot(np.linspace(15,35,20),mumu151515)
#plt.plot(np.linspace(15,35,20),mumu999)
#plt.show()