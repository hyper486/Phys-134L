import numpy as np
import matplotlib.pyplot as plt


# Wavelength array (nm)
wavelengths = np.linspace(400, 700, 301)

# Example QE curve (dimensionless)
QE_array = 0.8 * np.exp(-((wavelengths - 500)**2)/(2*50**2)) 

# Example B filter transmission curve (dimensionless)
T_B = 0.9 * np.exp(-((wavelengths - 440)**2)/(2*20**2)) 

# Example V filter transmission curve (dimensionless)
T_V = 0.9 * np.exp(-((wavelengths - 550)**2)/(2*20**2)) 

# Example source flux spectrum (photons s^-1 cm^-2 nm^-1)
F_lambda = 0.03 * np.ones_like(wavelengths)  # flat source for demonstration


def effective_qe(QE, T_filter, F_lam, lam):
    numerator = np.trapz(QE * T_filter * F_lam, x=lam)
    denominator = np.trapz(F_lam * T_filter, x=lam)
    return numerator / denominator if denominator != 0 else 0.0

QE_eff_B = effective_qe(QE_array, T_B, F_lambda, wavelengths)
QE_eff_V = effective_qe(QE_array, T_V, F_lambda, wavelengths)

print(f"Effective QE for B-band: {QE_eff_B:.3f}")
print(f"Effective QE for V-band: {QE_eff_V:.3f}")


# Telescope and detector parameters:
D = 0.35      # diameter in meters
D_cm = D * 100.0
A = np.pi * (D_cm/2)**2  # cm^2
epsilon = 0.5

# Detector parameters:
N_R   = 3.0      # Read noise (e-)
i_DC  = 0.01       # Dark current (e-/pix-s)
F_beta = 1e-2     # Background flux (photons s^-1 cm^-2 arcsec^-2)
pixel_scale = 0.74 
Omega = (pixel_scale**2) # arcsec^2/pixel

# Integrate the source flux over the filter band to get F (photons s^-1 cm^-2)
def band_flux(F_lam, T_filter, lam):
    # Integrate over the band and divide by the bandwidth for a representative flux
    num = np.trapz(F_lam * T_filter, x=lam)
    width = lam[-1] - lam[0]
    return num / width if width != 0 else 0.0

F_B = band_flux(F_lambda, T_B, wavelengths)
F_V = band_flux(F_lambda, T_V, wavelengths)

# Compute effective area for each filter:
# A_ε = A * ε * QE_eff
A_e_B = A * epsilon * QE_eff_B
A_e_V = A * epsilon * QE_eff_V


# S/N = (F A_ε sqrt(τ)) / sqrt((N_R²/τ) + F A_ε + i_DC + F_β A_ε Ω)
def snr(F, A_e, tau, N_R, i_DC, F_beta, Omega):
    numerator = F * A_e * np.sqrt(tau)
    denominator = np.sqrt((N_R**2 / tau) + F * A_e + i_DC + (F_beta * A_e * Omega))
    return numerator / denominator

# Integration times:
integration_times = np.array([1, 3, 5, 10, 30, 60, 100, 300, 600, 1000])

# Compute S/N for B and V
snr_B = [snr(F_B, A_e_B, t, N_R, i_DC, F_beta, Omega) for t in integration_times]
snr_V = [snr(F_V, A_e_V, t, N_R, i_DC, F_beta, Omega) for t in integration_times]

# Print results
print("\nB-band Results:")
print("Integration Time (s) | S/N")
print("---------------------|---------")
for t, val in zip(integration_times, snr_B):
    print(f"{t:>19} | {val:.3f}")

print("\nV-band Results:")
print("Integration Time (s) | S/N")
print("---------------------|---------")
for t, val in zip(integration_times, snr_V):
    print(f"{t:>19} | {val:.3f}")

# Plot results
plt.figure(figsize=(8,6))
plt.plot(integration_times, snr_B, marker='o', label='B-band')
plt.plot(integration_times, snr_V, marker='s', label='V-band')
plt.xlabel('Integration Time (s)', fontsize=14)
plt.ylabel('S/N', fontsize=14)
plt.title('S/N vs Integration Time for B and V filters', fontsize=16)
plt.grid(True)
plt.xscale('log')
plt.legend()
plt.show()
