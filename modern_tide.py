import math
import sys
import numpy as np
from configparser import ConfigParser
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  # optional, if needed for post-processing


# Read configuration file
parser = ConfigParser()
input_folder = input('Enter the name of the IC folder ')
parser.read(input_folder + '/const2ba.txt')

# Read physical variables
L1 = parser.getfloat('physical_variables', 'L1')
L2 = parser.getfloat('physical_variables', 'L2')
L3 = parser.getfloat('physical_variables', 'L3')
L4 = parser.getfloat('physical_variables', 'L4')
L5 = parser.getfloat('physical_variables', 'L5')
L6 = parser.getfloat('physical_variables', 'L6')
L7 = parser.getfloat('physical_variables', 'L7')

D1 = parser.getfloat('physical_variables', 'D1')
D2 = parser.getfloat('physical_variables', 'D2')
D3 = parser.getfloat('physical_variables', 'D3')
D4 = parser.getfloat('physical_variables', 'D4')
D5 = parser.getfloat('physical_variables', 'D5')
D6 = parser.getfloat('physical_variables', 'D6')
D7 = parser.getfloat('physical_variables', 'D7')

B1 = parser.getfloat('physical_variables', 'B1')
B2 = parser.getfloat('physical_variables', 'B2')
B3 = parser.getfloat('physical_variables', 'B3')
B4 = parser.getfloat('physical_variables', 'B4')
B5 = parser.getfloat('physical_variables', 'B5')
B6 = parser.getfloat('physical_variables', 'B6')
B7 = parser.getfloat('physical_variables', 'B7')

S   = parser.getfloat('physical_variables', 'S')
Sin = parser.getfloat('physical_variables', 'Sin')
a0  = parser.getfloat('physical_variables', 'a0')
nu  = parser.getfloat('physical_variables', 'nu')
g   = parser.getfloat('physical_variables', 'g', fallback=9.81)

kse1 = parser.getfloat('physical_variables', 'kse1', fallback=30)
kse2 = parser.getfloat('physical_variables', 'kse2', fallback=30)
kse3 = parser.getfloat('physical_variables', 'kse3', fallback=30)
kse4 = parser.getfloat('physical_variables', 'kse4', fallback=30)
kse5 = parser.getfloat('physical_variables', 'kse5', fallback=30)
kse6 = parser.getfloat('physical_variables', 'kse6', fallback=30)
kse7 = parser.getfloat('physical_variables', 'kse7', fallback=30)

U10 = parser.getfloat('physical_variables', 'U10', fallback=0)
U20 = parser.getfloat('physical_variables', 'U20', fallback=0)
U30 = parser.getfloat('physical_variables', 'U30', fallback=0)
U40 = parser.getfloat('physical_variables', 'U40', fallback=0)
U50 = parser.getfloat('physical_variables', 'U50', fallback=0)
U60 = parser.getfloat('physical_variables', 'U60', fallback=0)
U70 = parser.getfloat('physical_variables', 'U70', fallback=0)
h20 = parser.getfloat('physical_variables', 'h20', fallback=0)
hin0 = parser.getfloat('physical_variables', 'hin0', fallback=0)

threshold_h2 = parser.getfloat('numerical_variables', 'acn', fallback=0.001)
threshold_hin = parser.getfloat('numerical_variables', 'acs', fallback=0.001)


digitsn = math.ceil(-math.log10(threshold_h2))
digitss = math.ceil(-math.log10(threshold_hin))

nt = parser.getfloat('numerical_variables', 'nt', fallback=10)
n = parser.getfloat('numerical_variables', 'n', fallback=2)

# Derived quantities
w = np.sqrt(g/((Sin)*(B5*D5/L5)+(B6*D6/L6)))
Q = w/nu

threshold_h2  = threshold_h2/a0 
threshold_hin = threshold_hin/a0  





dim = int(nt * n * w/nu) + 1
tmax = 2 * n * np.pi

# Preallocate arrays for output
hs   = np.zeros(dim)
hsin = np.zeros(dim)
hs1  = np.zeros(dim)
u1s  = np.zeros(dim)
u2s  = np.zeros(dim)
u3s  = np.zeros(dim)
u4s  = np.zeros(dim)
u5s  = np.zeros(dim)
u6s  = np.zeros(dim)
u7s  = np.zeros(dim)

# Initialize simulation variables
h2 = h20
hin = hin0
U1 = U10
U2 = U20
U3 = U30
U4 = U40
U5 = U50
U6 = U60
U7 = U70

dt = nu * 6.28/(nt * w)
print("dt =", dt)

# Create a uniform time array (in hours)
T = np.arange(0, dim, 1) * dt/(2*np.pi)

C1 = kse1 * pow(D1, 1/6) / np.sqrt(g)
C2 = kse2 * pow(D2, 1/6) / np.sqrt(g)
C3 = kse3 * pow(D3, 1/6) / np.sqrt(g)
C4 = kse4 * pow(D4, 1/6) / np.sqrt(g)
C5 = kse5 * pow(D5, 1/6) / np.sqrt(g)
C6 = kse6 * pow(D6, 1/6) / np.sqrt(g)
C7 = kse7 * pow(D7, 1/6) / np.sqrt(g)

U0 = (S + Sin) * nu * a0 / (B1*D1 + B2*D2 + B3*D3 + B4*D4 + B5*D5 + B6*D6)

d11 = nu * U0 * L1 / (g * a0)
d21 = U0**2 * L1 / (g * C1**2 * a0 * D1)
d12 = nu * U0 * L2 / (g * a0)
d22 = U0**2 * L2 / (g * C2**2 * a0 * D2)
d13 = nu * U0 * L3 / (g * a0)
d23 = U0**2 * L3 / (g * C3**2 * a0 * D3)
d14 = nu * U0 * L4 / (g * a0)
d24 = U0**2 * L4 / (g * C4**2 * a0 * D4)
d15 = nu * U0 * L5 / (g * a0)
d25 = U0**2 * L5 / (g * C5**2 * a0 * D5)
d16 = nu * U0 * L6 / (g * a0)
d26 = U0**2 * L6 / (g * C6**2 * a0 * D6)
d17 = nu * U0 * L7 / (g * a0)
d27 = U0**2 * L7 / (g * C7**2 * a0 * D7)


# Set up lists to record local peaks for h2 and hin
h2_peaks  = []
hin_peaks = []

# Variables to detect rising/falling for each series
increasing_h2  = True
increasing_hin = True
prev_h2  = h2
prev_hin = hin

# For max values after t>2 (if needed)
mas  = 0
mas2 = 0

i = 0
# Use a while loop that can break early when criteria are met
while i < dim:
    # Record current simulation values
    hs[i]   = h2
    hsin[i] = hin
    hs1[i]  = a0 * math.sin(i * dt)  # outer values
    u1s[i]  = U1
    u2s[i]  = U2
    u3s[i]  = U3
    u4s[i]  = U4
    u5s[i]  = U5
    u6s[i]  = U6
    u7s[i]  = U7

    # Update simulation variables
    k = 1  # (could be time-dependent)
    U1 += dt * ((-h2 + math.sin(k * i * dt)) / d11 - (d21/d11) * U1 * abs(U1))
    U2 += dt * ((-h2 + math.sin(k * i * dt)) / d12 - (d22/d12) * U2 * abs(U2))
    U3 += dt * ((-h2 + math.sin(k * i * dt)) / d13 - (d23/d13) * U3 * abs(U3))
    U4 += dt * ((-h2 + math.sin(k * i * dt)) / d14 - (d24/d14) * U4 * abs(U4))
    U5 += dt * ((-hin + math.sin(k * i * dt)) / d15 - (d25/d15) * U5 * abs(U5))
    U6 += dt * ((-hin + math.sin(k * i * dt)) / d16 - (d26/d16) * U6 * abs(U6))
    U7 += dt * ((hin - h2) / d17 - (d27/d17) * U7 * abs(U7))
    
    # Use the previous stored values to update h2 and hin
    h2 += dt * (u1s[i]*D1*B1 + u2s[i]*D2*B2 + u3s[i]*D3*B3 + u4s[i]*D4*B4 + u7s[i]*D7*B7) * U0 / (S * a0 * nu)
    hin += dt * (u5s[i]*D5*B5 + u6s[i]*D6*B6 - u7s[i]*D7*B7) * U0 / (Sin * a0 * nu)
    
    # Check for local maximum for h2
    if increasing_h2 and (h2 < prev_h2):
        h2_peaks.append(prev_h2)
        increasing_h2 = False
    elif not increasing_h2 and (h2 > prev_h2):
        increasing_h2 = True
    prev_h2 = h2

    # Check for local maximum for hin
    if increasing_hin and (hin < prev_hin):
        hin_peaks.append(prev_hin)
        increasing_hin = False
    elif not increasing_hin and (hin > prev_hin):
        increasing_hin = True
    prev_hin = hin

    # For maximum values after t > 2 hours (if needed)
    t = T[i]

    # Check if the stopping criteria are met:
    if len(h2_peaks) >= 2 and len(hin_peaks) >= 2:
        diff_h2  = abs(h2_peaks[-1] - h2_peaks[-2])
        diff_hin = abs(hin_peaks[-1] - hin_peaks[-2])
        if diff_h2 < threshold_h2 and diff_hin < threshold_hin:
            print("convergence is reached")
            break

    i += 1

if i == dim:
    print: print("WARNING: CONVERGENCE WASN'T REACHED, TRY A REASONABLE VALUE FOR ACCURACY OR INCREASE THE NUMBER OF POINTS nt")
    sys.exit()
# Truncate arrays to the number of steps executed if loop ended early.
hs   = hs[:i]
hsin = hsin[:i]
hs1  = hs1[:i]
u1s  = u1s[:i]
u2s  = u2s[:i]
u3s  = u3s[:i]
u4s  = u4s[:i]
u5s  = u5s[:i]
u6s  = u6s[:i]
u7s  = u7s[:i]
T    = T[:i]

# Scale results if needed
hs   = hs * a0
hsin = hsin * a0

# Save output data to files (ensure the directory 'output_data' exists)
np.savetxt('output_data/h', hs)
np.savetxt('output_data/u1', u1s)
np.savetxt('output_data/u2', u2s)


MTRs = 2*a0*np.array(h2_peaks)[-1]
MTRs = f"{np.array(MTRs):.{digitsn}f}"

MTRn = 2*a0*np.array(hin_peaks)[-1]
MTRn = f"{np.array(MTRn):.{digitss}f}"

print("Simulation completed")

print("MTR Northern Bay: ", MTRn)
print("MTR Southern Bay: ", MTRs)

# ------------------------ Plotting ------------------------
# Define indices for plotting a segment of the simulation (if enough points exist)
s_idx = int(i-(3*np.pi/dt))
t_idx = int(i)
if t_idx > len(T):
    t_idx = len(T)

plt.figure()
plt.plot(12.5*(T[s_idx:t_idx]-T[s_idx]), hs[s_idx:t_idx], linestyle='-', linewidth=1, color='r', label='southern inner values')
plt.plot(12.5*(T[s_idx:t_idx]-T[s_idx]), hsin[s_idx:t_idx], linestyle='-', linewidth=1, color='g', label='northern inner values')
plt.plot(12.5*(T[s_idx:t_idx]-T[s_idx]), hs1[s_idx:t_idx], linestyle='-', linewidth=1, color='b', label='outer values')
plt.axhline(y=0.30, color='gray', linestyle='--', label='MTR data 1939')
plt.axhline(y=-0.30, color='gray', linestyle='--')
plt.xlabel('time [h]')
plt.ylabel('h [m]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


