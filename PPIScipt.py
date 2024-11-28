import numpy as np
from scipy.sparse import diags, block_diag 
from scipy.sparse.linalg import spsolve 
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from matplotlib.animation import FuncAnimation

def Fu(u_0, dt , v_0,alpha):
    F = u_0 - dt *( (u_0*v_0)/(u_0 + alpha))
    F[0, :] = u_0[2, :]
    F[599, :] = u_0[597, :]
    F[:, 0] = u_0[:, 2]
    F[:, 599] =u_0[:, 597]
    F=F.flatten()
    return(F.T)

def Fv(u_0, dt , v_0,alpha):
    F = v_0 + dt * ((u_0*v_0)/(u_0 + alpha))
    F[0, :] = v_0[2, :]
    F[599, :] = v_0[597, :]
    F[:, 0] = v_0[:, 2]
    F[:, 599] =v_0[:, 597]
    F=F.flatten()
    return(F.T)

def Au(u0,d,dt,beta,Fu):
    uflatten = u0.flatten()
    modifiedu = 1 + 4 *d - dt * beta*(1-uflatten)
    blocks = [modifiedu[i:i+600] for i in range(0, len(modifiedu), 600)]
    diagonal_blocks = []
    for block in blocks:
        main_diag = block
        identity = np.ones(len(block) - 1)
        side_diag_upper = -d*identity
        side_diag_lower = -d*identity
        offsets = [0, 1, -1]
        block_matrix = diags([main_diag, side_diag_upper, side_diag_lower], offsets).toarray()
        diagonal_blocks.append(block_matrix)

    block_matrix = block_diag(diagonal_blocks)
    offset2 = [-600, 600]
    identity2 = -d*np.ones(360000-600)
    final_matrix = block_matrix + diags([identity2, identity2], offset2)
    u = spsolve(final_matrix,Fu)
    u = u.reshape((600, 600))
    return(u)


def Av(e,d,dt,gamma,Fv):
    v0 = np.ones(600*600)
    vflatten = v0.flatten()
    modifiedv = 1 + 4 *e +dt*gamma*vflatten
    blocks = [modifiedv[i:i+600] for i in range(0, len(modifiedv), 600)]
    diagonal_blocks = []
    for block in blocks:
        main_diag = block
        identity = np.ones(len(block) - 1)
        side_diag_upper = -e*identity
        side_diag_lower = -e*identity
        offsets = [0, 1, -1]
        block_matrix = diags([main_diag, side_diag_upper, side_diag_lower], offsets).toarray()
        diagonal_blocks.append(block_matrix)

    block_matrix = block_diag(diagonal_blocks)
    offset2 = [-600, 600]
    identity2 = -e*np.ones(360000-600)
    final_matrix = block_matrix + diags([identity2, identity2], offset2)
    uv = spsolve(final_matrix,Fv)
    uv = uv.reshape((600, 600))
    return(uv)


N, M = 300, 300 
dt = 1/384 
alpha = .5
beta = .01 
gamma = .5 
delte = 1 
d = 1/384/.5**2
e = (1/384 * delte)/.5**2
u0 = np.zeros((N,M))
v0 = np.zeros((N,M))
x = np.arange(0, 300, .5)
y = np.arange(0, 300, .5)

X,Y = np.meshgrid(x,y)

radius = 75
center_x, center_y = 150,150 
distancefromcenter = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

v_0 = np.where(distancefromcenter<= radius, .2, 0 )
u_0 = np.where(distancefromcenter<= radius, .2, 0 )

totalu = []
totalv = []
for x in range(10):
    F = Fu(u_0, dt , v_0,alpha)
    fv = Fv(u_0, dt , v_0,alpha)
    newu = Au(u_0,d,dt,beta,F)
    newv = Av(e,d,dt,gamma,fv)
    u_0 = newu
    v_0 = newv 
    totalu.append(np.sum(newu))
    totalv.append(np.sum(newv))


plt.figure(figsize=(10, 8))

# Subplot 1: Prey Density at Time
plt.subplot(2, 2, 1)
plt.imshow(u_0, aspect='equal')
plt.title('Prey Density at Time')
plt.colorbar()

# Subplot 2: Predator Density at Time
plt.subplot(2, 2, 2)
plt.imshow(v_0, aspect='equal')
plt.title('Predator Density at Time')
plt.colorbar()


# Subplot 3: Population of u vs. Population of v Over Time
plt.subplot(2, 2, (3, 4))
plt.plot(totalu, totalv, 'r-o')  # 'r-o' stands for red color, line, and circle markers
plt.xlabel('Total u')
plt.ylabel('Total v')
plt.title('Population of u vs. Population of v Over Time')
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()





