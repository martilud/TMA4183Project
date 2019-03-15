import numpy as np
import numpy.linalg
import scipy.sparse
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D

# Beam parameters
height = 5 
width = 10
k = 1.172e-5 # m^2/s https://en.wikipedia.org/wiki/Thermal_diffusivity

# Discretization, defined as number of discrete points, so there are (nx-2) * (ny-2) interior points
nx = 51
ny = 51
hx = width/(nx-1)
hy = height/(ny-1)

gam1start = np.floor(nx * 0.4)
gam1stop = np.ceil(nx * 0.6)

# Initialize
x = np.zeros((nx, ny))

y0 = np.zeros(nx)
x[:,ny-1] = y0

# Distributed control
def u(x1):
    if(x1>gam1start*hx and x1<gam1stop*hx):
        return 1
    else:
        return 0

alpha = 1
def distControl(x1,x2):
    return u(x1)*np.exp(-alpha * x2)

f = np.zeros((nx, ny))
for i in range(1,nx-2):
    for j in range(1,ny-2):
        f[i,j] = distControl((i-1)*hx, (j-1)*hy)
f[0, :]  = np.zeros(ny) 
f[nx-1, :] = np.zeros(ny)
f[:, 0]  = np.zeros(nx)
f[:, ny-1] = y0
plt.imshow(f)
plt.colorbar()
plt.show()


# Function that evaluates the resulting finite difference matrix
def A(x,nx,ny):
    #Assumes x is a nx times ny matrix
    out = np.array(x)
    out[0,:] = (x[1,:]-x[1,:])/hx
    out[nx-1] = (x[nx-2,:]-x[nx-1,:])/hx
    out[:,ny-1] = x[:,ny-1]
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            out[i,j] = - k*((x[i+1,j] - 2*x[i,j] + x[i-1,j])/(hy*hy) + (x[i,j+1] - 2*x[i,j] + x[i,j-1])/(hx*hx)) 
    return out

TOL = 0.001

# Conjuagate Gradient
r = f - A(x,nx,ny)
r0 = np.linalg.norm(r,'fro')
p = r
rr_next = np.linalg.norm(r,'fro')**2
for i in range(nx*ny):
    Ap = A(p,nx,ny)  # <-------
    rr_curr = rr_next
    beta = rr_curr/np.dot(p.reshape(nx*ny),Ap.reshape(nx*ny))
    x = x + beta*p
    r = r - beta*Ap
    if np.linalg.norm(r,'fro')/r0 < TOL:
        break
    rr_next = np.linalg.norm(r,'fro')**2
    gamma = rr_next/rr_curr
    p = r + gamma*p
    print(rr_curr)
plt.imshow(x, cmap = 'gray')
plt.savefig('blurred.pdf')
plt.show()

##Dxx = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx,nx))#.toarray()
##Dyy = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(ny,ny))#.toarray()
###L = np.kron(Dyy, np.eye(nx)) + np.kron(np.eye(ny), Dxx)
###A = hx*hy*np.eye(nx*ny) + alpha*L
###f = np.ones(n*n) * h*h
##f = image.reshape(nx*ny)
##print(f)
###u = np.linalg.solve(A,f)
###fig = plt.figure()
###ax = fig.gca(projection='3d')
###surf = ax.plot_surface(X,Y,u.reshape(n,n))
###plt.show()
