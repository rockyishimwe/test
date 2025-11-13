# =========================
# GPS Positioning Simulation
# =========================
# Author: Rocky + GPT-5
# Description:
# This script simulates a GPS receiver position estimation using pseudoranges
# from 4 satellites, then visualizes the result in 3D.
# =========================

import numpy as np
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------
# CONSTANTS
# ------------------------
c = 3e8                # Speed of light (m/s)
e2 = 6.694379e-3       # Earth's eccentricity squared (WGS-84)
a = 6378137.0          # Earth's equatorial radius (m)

# ------------------------
# SATELLITE POSITIONS (ECEF)
# ------------------------
P = np.array([
    [15600000.0,  7540000.0, 20140000.0],
    [18760000.0,  2750000.0, 18650000.0],
    [17600000.0, 14600000.0, 13400000.0],
    [19170000.0,  6100000.0, 18390000.0]
], dtype=float)

# ------------------------
# TRUE RECEIVER POSITION (for simulation)
# ------------------------
SIMULATE_TRUE = True
true_lat_deg = 10.0
true_lon_deg = 20.0
true_h = 100.0
true_dt = 5e-5
noise_std = 5.0  # pseudorange noise (meters)

# ------------------------
# FUNCTIONS
# ------------------------

def geodetic_to_ecef(lat_deg, lon_deg, h):
    """Convert geodetic coordinates to ECEF Cartesian coordinates."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + h) * math.sin(lat)
    return np.array([x, y, z], dtype=float)

def ecef_to_geodetic(x, y, z, max_iter=10, tol=1e-12):
    """Convert ECEF Cartesian coordinates to geodetic (lat, lon, h)."""
    lon = math.atan2(y, x)
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(max_iter):
        N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
        h = p / math.cos(lat) - N
        new_lat = math.atan2(z, p * (1 - e2 * (N / (N + h))))
        if abs(new_lat - lat) < tol:
            lat = new_lat
            break
        lat = new_lat
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    h = p / math.cos(lat) - N
    return lat, lon, h

def haversine_m(lat1_deg, lon1_deg, lat2_deg, lon2_deg, R=6371000.0):
    """Compute surface distance between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1_deg, lon1_deg, lat2_deg, lon2_deg))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a_h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a_h))

# ------------------------
# SIMULATION: GENERATE PSEUDORANGES
# ------------------------
if SIMULATE_TRUE:
    x_true, y_true, z_true = geodetic_to_ecef(true_lat_deg, true_lon_deg, true_h)
    ranges = np.linalg.norm(P - np.array([x_true, y_true, z_true]), axis=1)
    rho = ranges + c * true_dt + np.random.normal(scale=noise_std, size=ranges.shape)
else:
    # manual pseudoranges (if SIMULATE_TRUE = False)
    rho = np.array([20432500.0, 21145000.0, 19760000.0, 21075000.0], dtype=float)

# ------------------------
# EQUATIONS TO SOLVE
# ------------------------
def equations(vars):
    x, y, z, dt = vars
    res = []
    for i in range(P.shape[0]):
        xi, yi, zi = P[i]
        dist = math.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2)
        res.append(dist + c * dt - float(rho[i]))
    return res

# ------------------------
# SOLVE FOR RECEIVER POSITION
# ------------------------
x0 = np.array([0.0, 0.0, 6371000.0, 0.0])
sol, infodict, ier, mesg = fsolve(equations, x0, full_output=True, maxfev=10000)

if ier != 1:
    print(f"Warning: solver did not converge ({mesg})")

x_est, y_est, z_est, dt_est = sol

# ------------------------
# CONVERT TO GEODETIC
# ------------------------
lat_rad, lon_rad, h_est = ecef_to_geodetic(x_est, y_est, z_est)
lat_deg = math.degrees(lat_rad)
lon_deg = math.degrees(lon_rad)

# ------------------------
# DISPLAY RESULTS
# ------------------------
print("\n=== ESTIMATED RECEIVER POSITION (ECEF) ===")
print(f"x = {x_est:,.3f} m")
print(f"y = {y_est:,.3f} m")
print(f"z = {z_est:,.3f} m")
print(f"Clock bias dt = {dt_est:.9e} s (range bias = {c*dt_est:,.3f} m)")

print("\n=== GEODETIC COORDINATES ===")
print(f"Latitude  = {lat_deg:.8f} °")
print(f"Longitude = {lon_deg:.8f} °")
print(f"Altitude  = {h_est:,.3f} m")

if SIMULATE_TRUE:
    ecef_err = np.linalg.norm(np.array([x_est, y_est, z_est]) - np.array([x_true, y_true, z_true]))
    hav = haversine_m(lat_deg, lon_deg, true_lat_deg, true_lon_deg)
    print("\n=== SIMULATION COMPARISON ===")
    print(f"True position: lat={true_lat_deg}°, lon={true_lon_deg}°, h={true_h} m")
    print(f"ECEF error = {ecef_err:,.3f} m")
    print(f"Surface (haversine) error = {hav:,.3f} m")
    print(f"Altitude error = {abs(h_est - true_h):.3f} m")

# ------------------------
# 3D VISUALIZATION
# ------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2], color='orange', label='Satellites', s=60)
ax.scatter(x_est, y_est, z_est, color='red', label='Estimated Receiver', s=80)
if SIMULATE_TRUE:
    ax.scatter(x_true, y_true, z_true, color='green', label='True Receiver', s=80)

# Draw lines between receiver and satellites
for sat in P:
    ax.plot([x_est, sat[0]], [y_est, sat[1]], [z_est, sat[2]], 'gray', linestyle='--', linewidth=0.8)

ax.set_title('GPS Trilateration Visualization')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.show()