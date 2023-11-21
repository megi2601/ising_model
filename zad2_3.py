import numpy as np 
from matplotlib import pyplot as plt
from numba import jit
from scipy.special import ellipe, ellipk


T = np.linspace(1, 5, 40)
Tc = 2 / np.log(1+np.sqrt(2))

M = np.where(T<Tc, (1-1/(np.sinh(2/T))**4)**0.125, 0)

def c(t):
    k1 = 2*np.tanh(2/t)/np.cosh(2/t)
    k2 = 2*(np.tanh(2/t))**2 - 1
    return 2/np.pi*((1/np.tanh(2/t)/t)**2)*(2*ellipk(k1*k1)-2*ellipe(k1*k1)-(1-k2)*(np.pi/2 + k2*ellipk(k1*k1)))

C = c(T)

@jit(nopython=True)
def custom_roll1(arr, shift):
    rows, cols = arr.shape
    shift %= cols  
    result = arr.copy()
    for i in range(rows):
        result[i] = np.roll(arr[i], shift)
    return result

@jit(nopython=True)
def custom_roll0(arr, shift):
    rows, cols = arr.shape
    shift %= rows
    result = arr.copy()
    for j in range(cols):
        result[:, j] = np.roll(arr[:, j], shift)
    return result


@jit(nopython=True)
def energia(lattice):
    E = custom_roll0(lattice, 1)
    E+= custom_roll0(lattice, -1)
    E+= custom_roll1(lattice, 1)
    E+= custom_roll1(lattice, -1)
    return np.sum(-lattice*E / 2)

@jit(nopython=True)
def MCstep(lattice, T):
    L = len(lattice)
    n = np.random.randint(0, L)
    m = np.random.randint(0, L)
    h = 0
    for pos in [(m, (n+1)%L), (m, (n-1)%L), ((m+1)%L, n), ((m-1)%L, n)]:
        h+=lattice[pos] 
    delta = 2*h
    r = np.random.rand()
    p = 1/(1+np.exp(-1/T*delta))
    if r<p:
        lattice[m, n] = 1
    else:
        lattice[m, n] = -1


steps = 5000

@jit(nopython=True)
def evolve(L, T):
    N = L*L
    lattice = np.ones((L, L))
    m = np.zeros(len(T))
    p = np.zeros(len(T))
    Q = np.zeros(len(T))
    for i in range(len(T)):
        temp = T[i]
        for _ in range(2000*N):
            MCstep(lattice, temp)

        magnet_matrix = np.zeros(steps)
        E_matrix = np.zeros(steps)
        for k in range(steps):
            
            for __ in range(N):
                MCstep(lattice, temp)

            magnet_matrix[k] = np.mean(lattice)
            E_matrix[k] = energia(lattice)
        m[i] = np.mean(np.abs(magnet_matrix))
        p[i] = np.var(magnet_matrix) / temp * N
        Q[i] = np.var(E_matrix) / temp / temp / N
        print(temp)
    res = [m, p, Q]
    return res

l10 = evolve(10, T)
l20 = evolve(20, T)

fig, ax = plt.subplots(2, figsize=(10, 10))

ax[0].plot(T, M, color='black', label="analitycznie")
ax[0].set_xlabel("Temperatura")
ax[1].set_xlabel("Temperatura")
ax[0].set_ylabel("Średni moduł z magnetyzacji")
ax[1].set_ylabel("Podatność magnetyczna")


ax[0].plot(T, l10[0], marker = "^", color = 'orange', label="L=10")
ax[1].plot(T, l10[1], marker = ".", color="orange",  label="L=10")


ax[0].plot(T, l20[0], marker = "^", color = 'red',  label="L=20")
ax[1].plot(T, l20[1], marker = ".", color="red",  label="L=20")

ax[0].legend()
ax[1].legend()
plt.savefig("wynik", dpi=200)

# plt.clf()

# l50 = evolve(50, T)

# plt.plot(T, C, color="black", label='analitycznie')
# plt.plot(T, l50[2], "o", color='orange', label="L=50")
# plt.legend()
# plt.xlabel("Temperatura")
# plt.ylabel("Ciepło właściwe (na spin)")
# plt.savefig("dodatkowe", dpi=200)    