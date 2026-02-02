#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
from astropy.table import Table
import psutil
import gc
import glob
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import time
from astropy.coordinates import SkyCoord
import astropy.units as u


# In[2]:
def determine_wavelength_grid(input_files,dr_name,galaxy_type):
    bands=['B','R','Z']
    # common wavelength grid (see paper)
    print('Determining wavelength range')
    #overlapping

    # initialize min and max redshift
    min_redshift = np.inf
    max_redshift = -np.inf
    
    # observed wavelength range (same for all)
    obs_wave_min = np.inf
    obs_wave_max = -np.inf

    print("Finding redshift range and combined wavelength coverage")
    for file in input_files:
        sky_fiber_table = Table.read(file, path=dr_name)
        
        # Get redshift range
        z_values = sky_fiber_table['NEAR_GALAXY_Z']
        min_redshift = min(min_redshift, np.min(z_values))
        max_redshift = max(max_redshift, np.max(z_values))

        # obs wavelength range (only done once for first fiber)
        if obs_wave_min == np.inf:
            sample_fiber = sky_fiber_table[0]

            for band in bands:
                wave_observed = sample_fiber[f'{band}_WAVELENGTH']
                mask = sample_fiber[f'{band}_MASK']
                valid = (mask == 0)
                if np.any(valid):
                    obs_wave_min = min(obs_wave_min, np.min(wave_observed[valid]))
                    obs_wave_max = max(obs_wave_max, np.max(wave_observed[valid]))
        
        del sky_fiber_table
        gc.collect()
    print(f"Redshift range: {min_redshift:.4f} to {max_redshift:.4f}")
    print(f"Observed wavelength range: {obs_wave_min:.1f} to {obs_wave_max:.1f} Å")

    # Rest-frame range for minimum redshift
    rest_min_at_zmin = obs_wave_min / (1 + min_redshift)
    rest_max_at_zmin = obs_wave_max / (1 + min_redshift)
    
    # Rest-frame range for maximum redshift  
    rest_min_at_zmax = obs_wave_min / (1 + max_redshift)
    rest_max_at_zmax = obs_wave_max / (1 + max_redshift)

    print(f"Rest-frame range at z_min ({min_redshift:.4f}): {rest_min_at_zmin:.1f} to {rest_max_at_zmin:.1f} Å")
    print(f"Rest-frame range at z_max ({max_redshift:.4f}): {rest_min_at_zmax:.1f} to {rest_max_at_zmax:.1f} Å")


    # find union
    shared_min = min(rest_min_at_zmin, rest_min_at_zmax)
    shared_max = max(rest_max_at_zmin, rest_max_at_zmax)

    print(f"Shared rest-frame range: {shared_min:.1f} to {shared_max:.1f} Å")

    wavelength_grid = np.arange(np.floor(shared_min), np.ceil(shared_max), 1)

    print(f"Final wavelength grid: {len(wavelength_grid)} bins from {wavelength_grid[0]} to {wavelength_grid[-1]} Å")

    return wavelength_grid

def print_memory(msg=""):
    """Print current memory usage"""
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3  
    print(f"{msg} Memory: {mem:.2f} GB")

# In[3]:

def process_data(fiber, z_gal, wavelength_grid):
    bands=['B','R','Z']
    """process one fiber data for all bands"""
    # initialize
    all_wave_rest = []
    all_flux = []
    all_ivar = []

    for band in bands:
        wave_observed = fiber[f'{band}_WAVELENGTH']
        # paper divides by 1.3
        flux = fiber[f'{band}_FLUX']/1.3
        ivar = fiber[f'{band}_IVAR']
        mask = fiber[f'{band}_MASK']

        # Only use good (unmasked, nonzero ivar) data
        valid = (mask == 0) & (ivar > 0)

        # restframe wavelengths for valid data
        wave_rest = wave_observed[valid] / (1 + z_gal)

        # check overlap
        in_range = (wave_rest >= wavelength_grid[0]) & (wave_rest <= wavelength_grid[-1])

        all_wave_rest.append(wave_rest[in_range])
        all_flux.append(flux[valid][in_range])
        all_ivar.append(ivar[valid][in_range])

    if not all_wave_rest:
        return None, None, None, False
    
    # Combine all bands
    wave_rest_combined = np.concatenate(all_wave_rest)
    flux_combined = np.concatenate(all_flux)
    ivar_combined = np.concatenate(all_ivar)

    del all_wave_rest,all_flux,all_ivar
    
    return wave_rest_combined, flux_combined, ivar_combined, True


def stack_spectra(input_files,dr_name,galaxy_type):
    bands=['B','R','Z']
    """ stack spectra """
    batch_size = 50
    print("Starting to stack spectra")
          
    wavelength_grid = determine_wavelength_grid(input_files,dr_name,galaxy_type)
    fiber_count = np.zeros_like(wavelength_grid, dtype=int)

    # Initialize arrays for stacking
    # numerator (ΣFω)
    weighted_flux_stack = np.zeros_like(wavelength_grid, dtype=float)
    # denominator (Σω)
    ivar_stack = np.zeros_like(wavelength_grid, dtype=float)
    # track spectra in each bin
    n_spectra = np.zeros_like(wavelength_grid, dtype=int)

    # stats 
    total_pixels_assigned = 0
    no_overlap = 0
    insufficient_data = 0
    processed_fibers = 0
    total_fibers = 0
    successful_fibers = 0

    for batch_start in range(0, len(input_files), batch_size):
        batch_end = min(batch_start + batch_size, len(input_files))
        batch_files = input_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(input_files)-1)//batch_size + 1}")
        print(f"Files {batch_start+1}-{batch_end} of {len(input_files)}")
        
        for i, file in enumerate(batch_files):
            file_index = batch_start + i
            print(f"\nProcessing file {file_index+1}/{len(input_files)}")
            
            print_memory("Before load")
            sky_fiber_table = Table.read(file, path=dr_name)
            print_memory("After load")

            # Get redshift range
            z_values = sky_fiber_table['NEAR_GALAXY_Z']
            if galaxy_type == 'BGS':
                valid_z = (0 <= z_values) & (z_values <= 0.5) 
            elif galaxy_type == 'LRG':
                valid_z = (0.3 <= z_values) & (z_values <= 1) 
            z_values = z_values[valid_z]

            if len(z_values) <= 0:
                print(f"No fibers in redshift range in {file}")
                continue
            
            total_fibers += len(z_values)
            chunk_size = 50  
            
            valid_indices = np.where(valid_z)[0]  
            for chunk_start in range(0, len(valid_indices), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(valid_indices))
                chunk_indices = valid_indices[chunk_start:chunk_end]
                
                if chunk_start % (chunk_size * 10) == 0:  # Progress every 10 chunks
                    print(f"  Processing fibers {chunk_start+1}-{min(chunk_end, len(valid_indices))} of {len(valid_indices)}")
                
                for idx in chunk_indices:
                    fiber = sky_fiber_table[idx]
                    z_gal = fiber['NEAR_GALAXY_Z']

                    # get distance between skyfiber and galaxy center
                    gal_ra = fiber['NEAR_GALAXY_RA']
                    gal_dec = fiber['NEAR_GALAXY_DEC']
                    target_ra = fiber['TARGET_RA']
                    target_dec = fiber['TARGET_DEC']

                    # get coord
                    gal_coords = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg)
                    sky_coords = SkyCoord(ra=gal_ra*u.deg, dec=gal_dec*u.deg)
                    # get distance
                    r = gal_coords.separation(sky_coords).to_value(u.arcsec)
                    # weight
                    distance_weight = (10/r)

                    wave_rest_combined, flux_combined, ivar_combined, success = process_data(
                                                                                fiber, z_gal, wavelength_grid)
                    processed_fibers += 1

                    if not success:
                        insufficient_data +=1
                        continue

                successful_fibers += 1
                total_pixels_assigned += len(wave_rest_combined)

                # bin pixel,stay in bounds
                bin_indices = np.searchsorted(wavelength_grid, wave_rest_combined, side='left')
                valid_bins_mask = (bin_indices >= 0) & (bin_indices < len(wavelength_grid))
                bin_indices = bin_indices[valid_bins_mask]
                flux_combined = flux_combined[valid_bins_mask]
                ivar_combined = ivar_combined[valid_bins_mask]

                if len(bin_indices) == 0:
                        no_overlap += 1
                        continue
                # count which bin it contributes to 
                unique_bins = np.unique(bin_indices)
                fiber_count[unique_bins] += 1
                

                # weighted flux (for every pixel)
                weights = ivar_combined * distance_weight
                # numerator
                weighted_flux_stack += np.bincount(bin_indices, weights=weights* flux_combined, minlength=len(wavelength_grid))
                # denominator 
                ivar_stack += np.bincount(bin_indices, weights=weights, minlength=len(wavelength_grid))
                # count pixels per bin
                n_spectra += np.bincount(bin_indices, minlength=len(wavelength_grid))
                
        del sky_fiber_table
        gc.collect()
            
        print_memory(f"After batch {batch_start//batch_size + 1}")
        
    print(f"Total pixels assigned: {total_pixels_assigned}")
    print(f"Skipped (no overlap): {no_overlap}")
    print(f"Skipped (insufficient data): {insufficient_data}")
    
    # Final stacked spectrum
    valid_bins = (ivar_stack > 0)
    stacked_flux = np.zeros_like(wavelength_grid, dtype=float)
    stacked_error = np.zeros_like(wavelength_grid, dtype=float)
    
    # Final equation
    stacked_flux[valid_bins] = weighted_flux_stack[valid_bins] / ivar_stack[valid_bins]
    # Error: 1/√ Σω
    stacked_error[valid_bins] = np.sqrt(1.0 / ivar_stack[valid_bins])

    # divide by aperture area (d = 1.5'')
    aperture_area = np.pi * (.75) * (.75)
    stacked_flux= stacked_flux/aperture_area
    stacked_error= stacked_error/aperture_area
    
    print(f"Processed {processed_fibers} fibers")
    print(f"Max pixels/bin: {n_spectra.max()}, Min pixels/bin (valid): {n_spectra[valid_bins].min()}")
    print(f"Valid bins: {np.sum(valid_bins)}/{len(valid_bins)}")
    if np.sum(valid_bins) > 0:
        print(f"Flux range: {np.min(stacked_flux[valid_bins])} to {np.max(stacked_flux[valid_bins])}")
        print(f"Wavelength range (valid): {wavelength_grid[valid_bins].min()} - {wavelength_grid[valid_bins].max()} Å")
    else:
        print("WARNING: No valid bins in final stack!")

    all_fiber_mask = (fiber_count == total_fibers)
    print(f"Bins with all {total_fibers} fibers(Number of pixels shared across all spectra): {np.sum(all_fiber_mask)}")
    print(f"Average number of pixels across all spectra: {total_pixels_assigned / successful_fibers}")


    return wavelength_grid, stacked_flux, stacked_error, n_spectra, valid_bins, all_fiber_mask, processed_fibers


# In[4]:
def save_results(wave, flux, error, residual, 
                 snr_narrow, snr_broad, valid_bins, 
                 n_spectra, continuum, noise, output_dir,
                 galaxy_type,dr_name,processed_fibers):
    """Save numerical results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results as numpy arrays
    results = {
        'wavelength': wave,
        'flux': flux,
        'error': error,
        'residual': residual,
        'snr_narrow': snr_narrow,
        'snr_broad': snr_broad,
        'valid_bins': valid_bins,
        'n_spectra': n_spectra,
        'continuum': continuum,
        'noise': noise
    }
    
    # Save as .npz file
    np.savez_compressed(os.path.join(output_dir, f'{galaxy_type}_{dr_name}_skyfiber_coadd_results.npz'), **results)

    # Also save a summary text file
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("Stacked Spectrum Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total processed skyfibers: {processed_fibers}\n")
        f.write(f"Total wavelength bins: {len(wave)}\n")
        f.write(f"Valid bins: {np.sum(valid_bins)}\n")
        f.write(f"Wavelength range: {wave[0]:.1f} - {wave[-1]:.1f} Å\n")
        if np.sum(valid_bins) > 0:
            f.write(f"Valid wavelength range: {wave[valid_bins].min():.1f} - {wave[valid_bins].max():.1f} Å\n")
            f.write(f"Flux range: {flux[valid_bins].min():.2e} - {flux[valid_bins].max():.2e}\n")
            f.write(f"Max spectra per bin: {n_spectra.max()}\n")
            f.write(f"Min spectra per valid bin: {n_spectra[valid_bins].min()}\n")
        f.write(f"\nResults saved to: {output_dir}/\n")
        f.write("Files created:\n")
        f.write("- stacked_spectrum_results.npz (all numerical data)\n")
        f.write("- analysis_summary.txt (this file)\n")
    
    print(f"Results saved to {output_dir}/")

# In[5]:


def find_lines_in_spectrum (wave, flux, valid_bins):
    """ SG filter to create a smoothed spectrum; G filter to convolve new spectrum """
    # window must be odd
    window_length=205
    if window_length % 2 == 0:
        window_length += 1
    
    # initialize smoothed, coadded spectrum (continuum)
    continuum = np.zeros_like(flux,dtype=np.float32)
    
    if np.sum(valid_bins) > window_length:
        # apply SG filter on valid flux at valid bin
        continuum[valid_bins] = savgol_filter(flux[valid_bins], window_length=window_length, polyorder=3)

    # convolve residual spectrum (only valid bins!)
    residual = flux - continuum
    residual_for_conv = np.where(valid_bins, residual, 0)
    
    smoothed_narrow = gaussian_filter1d(residual_for_conv, sigma=3)
    smoothed_broad = gaussian_filter1d(residual_for_conv, sigma=15)
    # @ invalid bins: smoothed values = NaN
    smoothed_narrow[~valid_bins] = np.nan
    smoothed_broad[~valid_bins] = np.nan
    
    return residual, smoothed_narrow, smoothed_broad, continuum


# In[6]:


def compute_SNR(residual, smoothed_narrow, smoothed_broad, valid_bins):
    """ estimate noise, find SNR """
    # NOTE: this part of script isnt really useful - pass it into REAL SNR script in vscode
    print("Computing SNR")
    bin_size=400
    step=200
    noise = np.full_like(residual, np.nan)

    for i in range(0,len(residual), step):
        start = i
        end = min(i + bin_size, len(residual))
        
        mask = valid_bins[start:end]
        if np.sum(mask) > 10:
            indices = np.where(mask)[0] + start
            window_data = residual[indices]
            # skip if NaN
            if not np.any(np.isnan(window_data)):
                # calculate noise
                noise_estimate = 0.5 * (scoreatpercentile(window_data, 84) - scoreatpercentile(window_data, 16))
                # assign noise to valid pixels in bin
                valid_in_window = valid_bins[start:end]
                noise[start:end][valid_in_window] = noise_estimate
            
    # initialize array to store SNR  
    snr_narrow = np.full_like(residual, np.nan)
    snr_broad = np.full_like(residual, np.nan)

    # Calculate SNR @ valid noise / valid bins, not NaN
    valid_for_snr = (noise > 0) & valid_bins & ~np.isnan(smoothed_narrow) & ~np.isnan(smoothed_broad)
    snr_narrow[valid_for_snr] = smoothed_narrow[valid_for_snr] / noise[valid_for_snr]
    snr_broad[valid_for_snr] = smoothed_broad[valid_for_snr] / noise[valid_for_snr]

    return snr_narrow, snr_broad, noise


# In[7]:


def get_invalid_wavelengths(wave, valid_bins):
    """Return list of wavelengths where data is invalid """
    
    invalid_wavelengths = wave[~valid_bins]
    print(f"Invalid wavelengths: {len(invalid_wavelengths)} bins")
    print(f"Wavelength ranges with no data: ")

    # invalid wavelengths
    invalid_indices = np.where(~valid_bins)[0]
    
    if len(invalid_indices) == 0:
        print("None - all data is valid!")
        return []
        
    # Group consecutive indices
    ranges = []
    start = invalid_indices[0]
    end = start
    
    for i in range(1, len(invalid_indices)):
        if invalid_indices[i] == invalid_indices[i-1] + 1:
            # Consecutive
            end = invalid_indices[i]
        else:
            # Gap
            ranges.append((start, end))
            start = invalid_indices[i]
            end = start
    
    # Don't forget the last range
    ranges.append((start, end))
    
    # Print the ranges
    for i, (start_idx, end_idx) in enumerate(ranges):
        start_wave = wave[start_idx]
        end_wave = wave[end_idx]
        length = end_idx - start_idx + 1
        print(f"  Range {i+1}: {start_wave} - {end_wave} Å ({length} bins)")
    
    return invalid_wavelengths


# In[98]:

def plot_all_fiber_mask(wave, all_fiber_mask, alpha=0.15, color='green'):
    """Safely plot fiber mask without index errors"""
    mask_len = min(len(wave)-1, len(all_fiber_mask))
    for i in range(mask_len):
        if i < len(all_fiber_mask) and all_fiber_mask[i]:
            # don't go out of bounds
            if i+1 < len(wave):
                plt.axvspan(wave[i], wave[i+1], alpha=alpha, color=color)

def plot(wave,flux,continuum,residual,
         smoothed_narrow, smoothed_broad,
         noise,snr_narrow,snr_broad,valid,
         all_fiber_mask,output_dir, ends = True):
    
    os.makedirs(output_dir, exist_ok=True)
        
    # og stacked spectrum (blue) and Savitzky–Golay smooth continuum (pink)
    plt.figure(figsize=(14, 6))
    plt.plot(wave, flux, label='Stacked Spectrum', color='blue', alpha=0.7)
    plt.plot(wave, continuum, label='Savitzky–Golay Smooth Continuum', color='red')
    plot_all_fiber_mask(wave, all_fiber_mask, alpha=0.15, color='green')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux[10^(-17) ergs/s/cm^2/Å/arcsec^2]')
    plt.title('Original stacked Spectrum')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(os.path.join(output_dir, f'spectrum_continuum.png'))
    plt.close()


    # Residual
    plt.figure(figsize=(14, 6))
    plt.plot(wave, residual, label='Residual (Flux - Continuum)', color='blue', alpha=0.6)
    plot_all_fiber_mask(wave, all_fiber_mask, alpha=0.15, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Rest Wavelength (Å)')
    plt.ylabel('Residual Flux [10^(-17) ergs/s/cm^2/Å/arcsec^2]')
    plt.title('Residual Spectrum (Average flux rescaled around zero)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'residual_spectrum.png'))
    plt.close()
    
    
    # Gaussian smoothed residuals
    plt.figure(figsize=(14, 6))
    plt.plot(wave, smoothed_narrow, label='Gaussian Smoothed Residual (Narrow)', color='purple')
    plt.plot(wave, smoothed_broad, label='Gaussian Smoothed Residual (Broad)', color='orange')
    plot_all_fiber_mask(wave, all_fiber_mask, alpha=0.15, color='green')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux[10^(-17) ergs/s/cm^2/arcsec^2]')
    plt.title('Gaussian Convolution and estimated error')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(os.path.join(output_dir, f'smoothed_noise.png'))
    plt.close()
    
    #SNR (fix)
    plt.figure(figsize=(14, 6))
    plt.plot(wave, snr_narrow, label='SNR (3 Å )', color='green', alpha = 0.7)
    plt.plot(wave, snr_broad, label='SNR (15 Å )', color='black', alpha = 0.7)
    plot_all_fiber_mask(wave, all_fiber_mask, alpha=0.15, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Rest Wavelength (Å)')
    plt.ylabel('SNR')
    plt.title('Signal-to-Noise Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'snr_spectrum.png'))
    plt.close()

    print(f"All plots saved to {output_dir}/")

# In[9]:

def main(dr_name,galaxy_type):
    print_memory("Start")
    start_time = time.time()


    input_dir = f"/pscratch/sd/a/aizaa/{galaxy_type}/skyfiber_{dr_name}"
    file_dir = os.path.join(input_dir, f'{galaxy_type}_skyfibers_*.h5')
    input_files = sorted(glob.glob(file_dir))
    print(f"Found {len(input_files)} files to combine")

    base_output_dir = "/pscratch/sd/a/aizaa/results/surface_brightness" 
    plot_dir = os.path.join(base_output_dir, f"{galaxy_type}/{dr_name}_skyfiber")
    
    print_memory("Before stacking")
    wave, flux, error, n_spectra, valid_bins, all_fiber_mask,processed_fibers = stack_spectra(input_files, dr_name,galaxy_type)
    print_memory("After stacking")

    # apply filters
    residual, smoothed_narrow, smoothed_broad, continuum = find_lines_in_spectrum(wave, flux, valid_bins)
    # calculate SNR
    snr_narrow, snr_broad, noise = compute_SNR(residual, smoothed_narrow, smoothed_broad, valid_bins)

    # invalid
    invalid_waves = get_invalid_wavelengths(wave, valid_bins)
    print_memory("after filters")
    
    print("Saving results...")
    save_results(wave, flux, error, residual, snr_narrow, snr_broad, valid_bins, 
                n_spectra, continuum, noise, base_output_dir, galaxy_type,dr_name,processed_fibers)
    print("Done with calculations!")
    print_memory("done with filters")
    # smoothed curves alone:
    smoothed_narrow = snr_narrow * noise
    smoothed_broad = snr_broad * noise

    plot(wave,flux,continuum,residual,smoothed_narrow, smoothed_broad,noise,snr_narrow,snr_broad,valid_bins,all_fiber_mask,plot_dir)
    print_memory("after plotting")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    print("Done plotting!")

# In[ ]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DESI sky fibers')
    parser.add_argument('dr_name')
    parser.add_argument('galaxy_type')
    args = parser.parse_args()
    main(dr_name=args.dr_name,galaxy_type = args.galaxy_type)
# In[ ]:
