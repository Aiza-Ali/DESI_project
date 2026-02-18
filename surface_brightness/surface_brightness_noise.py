import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import scoreatpercentile

def load_results(gal_type):
    # load saved arrays from coadding
    path = f"{gal_type}_loa_skyfiber_coadd_results.npz"
    results = np.load(path)

    wavelength = results['wavelength']
    residual = results['residual']
    flux = results['flux']
    continuum = results['continuum']
    valid_bins = results['valid_bins']
    noise = results['noise']
    snr_broad = results['snr_broad']

    valid_wave = wavelength[valid_bins]
    valid_residual = residual[valid_bins]
    valid_flux = flux[valid_bins]
    valid_continuum = continuum[valid_bins]
    valid_noise = noise[valid_bins]
    valid_snr_broad = snr_broad[valid_bins]

    return valid_noise,valid_wave,valid_residual,valid_flux,valid_continuum,valid_snr_broad

def plot(valid_wave,valid_residual,valid_noise,valid_flux,valid_continuum,valid_smoothed_broad,gal_type):
    plt.figure(figsize=(10, 5))
    # valid_noise is in 10^{-17}
    plt.plot(valid_wave, 5*valid_noise*1000, label='Residual Spectrum (Flux - Smoothed Continuum)', color='blue', alpha=0.7, linewidth=1)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel(r"Residual Flux [$10^{-20} erg/s/cm^2/arcsec^2$]")
    plt.title(f'Surface Brightness 5σ : {gal_type}')
    if gal_type == "LRG":
        plt.xlim(3000,6500)
        plt.ylim(0,5)
    else:
        plt.xlim(3000,8500)
        plt.ylim(0,10)
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_graph(gal_type):
    valid_noise,valid_wave,valid_residual,valid_flux,valid_continuum,valid_snr_broad = load_results(gal_type)
    valid_smoothed_broad = valid_snr_broad * valid_noise
    plot(valid_wave,valid_residual,valid_noise,valid_flux,valid_continuum,valid_smoothed_broad,gal_type)

create_graph('LRG')
create_graph('BGS')