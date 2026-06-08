import math
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser

#read from the input file const4bo.txt

parser = ConfigParser()
input_folder = input('Enter the name of the IC folder ')
parser.read(input_folder+'/const4bo.txt')

threshold_h = parser.getfloat('numerical_variables', 'ac')

L1 = parser.getfloat('physical_variables', 'L1') 

L2 = parser.getfloat('physical_variables', 'L2') 

L3 = parser.getfloat('physical_variables', 'L3') 

L4 = parser.getfloat('physical_variables', 'L4') 

D1 = parser.getfloat('physical_variables', 'D1') 

D2 = parser.getfloat('physical_variables', 'D2')

D3 = parser.getfloat('physical_variables', 'D3')

D4 = parser.getfloat('physical_variables', 'D4')

B1 = parser.getfloat('physical_variables', 'B1') 

B2 = parser.getfloat('physical_variables', 'B2') 

B3 = parser.getfloat('physical_variables', 'B3')

B4 = parser.getfloat('physical_variables', 'B4') 

S = parser.getfloat('physical_variables', 'S')

a0 = parser.getfloat('physical_variables', 'a0') 

nu = parser.getfloat('physical_variables', 'nu') 

g = parser.getfloat('physical_variables', 'g',
                         fallback = 9.81) 
kse = parser.getfloat('physical_variables', 'kse',
                         fallback = 30) 
U10 = parser.getfloat('physical_variables', 'U10', 
                         fallback = 0) 
U20 = parser.getfloat('physical_variables', 'U20', 
                         fallback = 0) 
U30 = parser.getfloat('physical_variables', 'U30', 
                         fallback = 0) 
U40 = parser.getfloat('physical_variables', 'U40', 
                         fallback = 0) 
h20 = parser.getfloat('physical_variables', 'h20', 
                         fallback = 0) 

nt = parser.getfloat('numerical_variables', 'nt', 
                         fallback = 10) 

n = parser.getfloat('numerical_variables', 'n', 
                         fallback = 2) 



w = np.sqrt(g/S*(B1*D1/L1+B2*D2/L2+B3*D3/L3+B4*D4/L4))

Q = w/nu
print(Q)

dim = int(nt*n*w/nu)+1
tmax = 2*n*np.pi
hs1 = np.zeros(dim)
hsw = np.zeros(dim)
hs = np.zeros(dim)
u1s = np.zeros(dim)
u2s = np.zeros(dim)
u3s = np.zeros(dim)
u4s = np.zeros(dim)

hsi = np.zeros(dim)
u1si = np.zeros(dim)
u2si = np.zeros(dim)
u3si = np.zeros(dim)
u4si = np.zeros(dim)

delta = np.zeros(dim)
delta1 = np.zeros(dim)
delta2 = np.zeros(dim)

h2 = h20
U1 = U10
U2 = U20
U3 = U30
U4 = U40

dt = nu*6.28/(nt*w)
print(dt)

# Create a uniform array using NumPy
T = np.arange(0, dim, 1)*dt/(2*np.pi)



C1 = kse*pow(D1,1/6)/np.sqrt(g)
C2 = kse*pow(D2,1/6)/np.sqrt(g)
C3 = kse*pow(D3,1/6)/np.sqrt(g)
C4 = kse*pow(D4,1/6)/np.sqrt(g)

U0 = S*nu*a0/(B1*D1+B2*D2+B3*D3+B4*D4)
d11 = nu*U0*L1/(g*a0)
d21 = U0*U0*L1/(g*C1*C1*a0*D1)


d12 = nu*U0*L2/(g*a0)
d22 = U0*U0*L2/(g*C2*C2*a0*D2)

d13 = nu*U0*L3/(g*a0)
d23 = U0*U0*L3/(g*C3*C3*a0*D3)

d14 = nu*U0*L4/(g*a0)
d24 = U0*U0*L4/(g*C4*C4*a0*D4)


digits = math.ceil(-math.log10(threshold_h))
# Set up lists to record local peaks for h2 and hin
h_peaks  = []

# Variables to detect rising/falling for each series
increasing_h  = True
prev_h  = h2


t = 0
i = 0

#print(C1,C2,U0,d11,d12,d21,d22,a0,w)

U1i = U1
U2i = U2
U3i = U3
U4i = U4
h2i = h2

tau = 2*d21+d22/d21*d22
mas = 0
mnm = 0
while i<dim:
    hsw[i]= h20*math.cos(w*i*dt/nu)

    hs1[i]= math.sin(i*dt)
    delta2[i] =h20*(1-math.exp(-np.sqrt(dt*i/(50*tau))))
    hs[i]=h2
    u1s[i]=U1
    u2s[i]=U2
    u3s[i]=U3
    u4s[i]=U4
    k=1
    U1+= dt*((-h2+math.sin(k*i*dt))/d11-(d21/d11)*U1*abs(U1))
    U2+= dt*((-h2+math.sin(k*i*dt))/d12-(d22/d12)*U2*abs(U2))
    U3+= dt*((-h2+math.sin(k*i*dt))/d13-(d23/d13)*U3*abs(U3))
    U4+= dt*((-h2+math.sin(k*i*dt))/d14-(d24/d14)*U4*abs(U4))
    h2+= dt*(u1s[i]*D1*B1+u2s[i]*D2*B2+u3s[i]*D3*B3+u4s[i]*D4*B4)*U0/(S*a0*nu)

   # Check for local maximum for h2
    if increasing_h and (h2 < prev_h):
       h_peaks.append(prev_h)
       print(prev_h,"     ",i)
       increasing_h = False
    elif not increasing_h and (h2 > prev_h):
       increasing_h = True
    prev_h = h2

    t = T[i]
    # Check if the stopping criteria are met:
    if len(h_peaks) >= 2:
        diff_h  = abs(h_peaks[-1] - h_peaks[-2])
        if diff_h < threshold_h:
            print("convergence is reached")
            break
    i+=1

if i == dim:
    print("WARNING: CONVERGENCE WASN'T REACHED, TRY REASONABLE VALUES OF ACCURACY OR INCREASE THE NUMBER OF POINTS")
#/Users/Enrico/Desktop
#write on file
#delta = hs-hsi

hsi = a0*hsi
hs1 = a0*hs1
hs = a0*hs #+ hsw
delta = hsi-hs1
delta1 = hs-hsi
mas = round(a0*mas,15)
mnm = round(a0*mnm,15)
np.savetxt('output_data/h', hs)
np.savetxt('output_data/u1', u1s)
np.savetxt('output_data/u2', u2s)
MTR = 2*a0*np.array(h_peaks)[-1]
MTR = f"{np.array(MTR):.{digits}f}"

print("simulation completed")
print("Bay MTR = ", MTR, " m")

# Plotting the graph
fig = plt.gcf()
#fig.set_size_inches(18.5, 10.5)
fig.set_dpi(500)

s = int(i-3*np.pi/dt)
t = int(i)

plt.plot(12.5*(T[s:t]-T[s]), hs[s:t], marker='', linestyle='-',linewidth=1, color='r', label='inner values')  # Adjust styling as needed
plt.plot(12.5*(T[s:t]-T[s]), hs1[s:t], marker='', linestyle='-',linewidth=1, color='b', label='outer values')
plt.axhline(y=0.17, color='gray', linestyle='--', label='MTR data 1902')
plt.axhline(y=-0.17, color='gray', linestyle='--', label=None)
# Adding labels and title
plt.xlabel('time [h]')
plt.ylabel('h[m]')
#plt.title('Basin height')
#plt.savefig('output_data/h5difnum')
# Adding a legend (optional)

# Display the graph
plt.show()






