using PyPlot
using LinearAlgebra

# Beam parameters
height = 5 
width = 10
k = 1.172e-5 # m^2/s https://en.wikipedia.org/wiki/Thermal_diffusivity

# Discretization, defined as number of discrete points, so there are (nx-2) * (ny-2) interior points
nx = 5
ny = 5
hx = width/(nx-1)
hy = height/(ny-1)

gam1start = floor(nx * 0.3)
gam1stop = ceil(nx * 0.7)

# Initialize
x = zeros(nx, ny)

y0 = zeros(nx)
x[:,ny] = y0
# Distributed control
function u(x1)
    if(x1>gam1start*hx && x1<gam1stop*hx)
        return 1
    else
        return 0
    end
end

alpha = 1
function distControl(x1,x2)
    return u(x1)*exp(-alpha * x2)
end

f = Array{Float64}(undef, nx, ny)
for i=2:(nx-1)
    for j=2:(ny-1)
        f[i,j] = distControl((i-1)*hx, (j-1)*hy)
    end
end
f[1, :]  = zeros(ny) 
f[nx, :] = zeros(ny)
f[:, 1]  = zeros(nx)
f[:, ny] = y0




imshow(f, cmap = "inferno")
colorbar()
show()

print(reshape(f,nx*ny))

function applyA(x)                   
    out = Array{Float64}(undef,nx,ny)
    # Boundaries
    out[1, 2:ny] = (x[2,2:ny] - x[1,2:ny])/hx 
    out[nx, 2:ny] = (x[nx-1,2:ny] - x[nx,2:ny])/hx
    out[2:(nx-1), 1]= (x[2:(nx-1),2] - x[2:(nx-1),1])/hy
    out[:, ny] = x[:,ny]
    out[1,1] = 0; out[1,nx] = 0
    # Interior
    for i=2:(nx-1)
        for j=2:(ny-1)
            out[i,j] = - k*((x[i+1,j] - 2*x[i,j] + x[i-1,j])/(hx*hx) + (x[i,j+1] - 2*x[i,j] + x[i,j-1])/(hy*hy))
        end
    end
    return out
end


function beta(x1)
    return u(x1)
end

#Initial residual r = b- Ax
TOL = 1e-5
function CG(x)
    r = f - applyA(x)
    p = r
    imshow(r)
    colorbar()
    show()
    #rr_next = norm(r,2)^2
    rr_next = dot(reshape(r,nx*ny),reshape(r,nx*ny))
    for i = 1:(nx*ny)
        Ap = applyA(p)
        rr_curr = rr_next
        beta = rr_curr/(dot(reshape(p,nx*ny),reshape(Ap,nx*ny)))
        x = x + beta*p
        r_next = r - beta*Ap
        #rr_next = norm(r_next,2)
        rr_next = dot(reshape(r_next,nx*ny),reshape(r_next,nx*ny))
        if (rr_next < TOL)
            return x
        end
        gamma = rr_next/rr_curr
        p = r_next + gamma * p
        r = r_next
        print("Residual:", rr_next, "\n")
    end
    return x
end
imshow(transpose(CG(x)))
colorbar()
show()
