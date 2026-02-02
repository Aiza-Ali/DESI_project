#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import time
import numpy as np
from astropy.table import Table
import psutil
import gc
import glob
import argparse
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import redrock
from redrock.templates import load_templates_from_header
from astropy.io import fits
from astropy.io.fits import Header
import desispec
from desispec.resolution import Resolution
from desispec.interpolation import resample_flux

# In[2]
def print_memory(msg=""):
    """Print current memory usage"""
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3  
    print(f"{msg} Memory: {mem:.2f} GB")

# load templates
def load_templates(data):
    """ Read in all templates"""
    if data == "loa":
        templates = dict()
        for filename in redrock.templates.find_templates():
            t = redrock.templates.Template(filename)
            templates[(t.template_type, t.sub_type)] = t

    if data == "iron":
        header = Header()
        header['TEMNAM00']= 'GALAXY'
        header['TEMVER00'] = '2.6'
        templates = redrock.templates.load_templates_from_header(header)
        
    return templates

# In[3]
def compute_residual_spectrum(galaxy, data, templates, band):
    """ Compute/Return residual spectrum for a single galaxy in a specific band """
    # get wavelength, flux, redshift
    obs_wavelength = galaxy[f'{band}_WAVELENGTH']
    obs_flux = galaxy[f'{band}_FLUX']
    redshift = galaxy['Z']
    
    # Convert to rest frame 
    rest_wavelength = obs_wavelength / (1 + redshift)
    
    # Template type
    spectype = 'GALAXY'
    fulltype = (spectype, '')

    # Construct composite template spectrum
    if (data == 'iron'):
        template = templates[0]
        ncoeff = template.flux.shape[0] # num of template components
        coeff = galaxy['COEFF'] #weight for each component
        tflux = template.flux.T.dot(coeff) #weighted sum of components
    elif (data == 'loa'):
        ncoeff = templates[fulltype].flux.shape[0] # num of template components
        coeff = galaxy['COEFF'][0 : ncoeff] #weight for each component
        tflux = templates[fulltype].flux.T.dot(coeff) #weighted sum of components
    
    # redshift template to obs frame
    if (data == 'iron'):
        twave = template.wave * (1 + redshift)
    elif (data == 'loa'):
        twave = templates[fulltype].wave * (1 + redshift)

    # Spectral data resolution and flux
    R = Resolution(galaxy[f'{band}_RESOLUTION']) #matches spectograph resolution
    txflux = R.dot(resample_flux(rest_wavelength, twave, tflux)) # allign wavelength grid
    
    # Calculate residual flux 
    residual = obs_flux - txflux
    
        
    return residual, rest_wavelength, txflux

# In [4]
def find_lines_in_spectrum (wave, flux, valid_bins, window_length=205):
    """ SG filter to create a smoothed spectrum; G filter to convolve new spectrum """
    # window must be odd
    if window_length % 2 == 0:
        window_length += 1
    
    # initialize smoothed, coadded spectrum (continuum)
    continuum = np.zeros_like(flux)
    
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

def compute_SNR(residual, smoothed_narrow, smoothed_broad, valid_bins, bin_size=400, step=200):
    """ estimate noise, find SNR """
    print("Computing SNR")
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

# In [5]
def plot_all_galaxy_mask(wave, all_galaxy_mask, alpha=0.15, color='green'):
    """Safely plot galaxy mask without index errors"""
    mask_len = min(len(wave)-1, len(all_galaxy_mask))
    for i in range(mask_len):
        if i < len(all_galaxy_mask) and all_galaxy_mask[i]:
            # don't go out of bounds
            if i+1 < len(wave):
                plt.axvspan(wave[i], wave[i+1], alpha=alpha, color=color)

def plot(wave,flux,continuum,residual,
         smoothed_narrow, smoothed_broad,
         noise,snr_narrow,snr_broad,valid,
         all_galaxy_mask,output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # og stacked spectrum (blue) and Savitzky–Golay smooth continuum (pink)
    plt.figure(figsize=(14, 6))
    plt.plot(wave, flux, label='Stacked Spectrum', color='blue', alpha=0.7)
    plt.plot(wave, continuum, label='Savitzky–Golay Smooth Continuum', color='red')
    plot_all_galaxy_mask(wave, all_galaxy_mask, alpha=0.15, color='green')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux[10^(-17) ergs/s/cm^2/Å]')
    plt.title('Original stacked Spectrum (green = all skygalaxys)')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(os.path.join(output_dir, f'spectrum_continuum.png'))
    plt.close()


    # Residual
    plt.figure(figsize=(14, 6))
    plt.plot(wave, residual, label='Residual (Flux - Continuum)', color='blue', alpha=0.6)
    plot_all_galaxy_mask(wave, all_galaxy_mask, alpha=0.15, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Rest Wavelength (Å)')
    plt.ylabel('Residual Flux [10^(-17) ergs/s/cm^2/Å]')
    plt.title('Residual Spectrum (Average flux rescaled around zero) (green = all skygalaxys)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'residual_spectrum.png'))
    plt.close()
    
    
    # Gaussian smoothed residuals and error
    plt.figure(figsize=(14, 6))
    plt.plot(wave, smoothed_narrow, label='Gaussian Smoothed Residual (Narrow)', color='purple')
    plt.plot(wave, smoothed_broad, label='Gaussian Smoothed Residual (Broad)', color='orange')
    plot_all_galaxy_mask(wave, all_galaxy_mask, alpha=0.15, color='green')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux[10^(-17) ergs/s/cm^2]')
    plt.title('Gaussian Convolution and estimated error (green = all skygalaxys)')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(os.path.join(output_dir, f'smoothed_noise.png'))
    plt.close()
    
    #SNR
    plt.figure(figsize=(14, 6))
    plt.plot(wave, snr_narrow, label='SNR (3 Å )', color='green', alpha = 0.7)
    plt.plot(wave, snr_broad, label='SNR (15 Å )', color='black', alpha = 0.7)
    plot_all_galaxy_mask(wave, all_galaxy_mask, alpha=0.15, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Rest Wavelength (Å)')
    plt.ylabel('SNR')
    plt.title('Signal-to-Noise Ratio (green = all skygalaxys)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'snr_spectrum.png'))
    plt.close()

    print(f"All plots saved to {output_dir}/")

# In[6]:
def save_results(wave, flux, error, residual, 
                 snr_narrow, snr_broad, valid_bins, 
                 n_spectra, continuum, noise, output_dir,
                 galaxy_type,dr_name,processed_galaxies):
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
    np.savez_compressed(os.path.join(output_dir, f'{galaxy_type}_{dr_name}_galaxy_coadd_results.npz'), **results)
    
    # Also save a summary text file
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("Stacked Spectrum Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
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

# In[7]:
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
        z_values = sky_fiber_table['Z']
        # if galaxy_type == 'BGS':
        #     valid_z = (0 <= z_values) & (z_values <= 0.5) 
        # elif galaxy_type == 'LRG':
        #     valid_z = (0.3 <= z_values) & (z_values <= 1) 
        # z_values = z_values[valid_z]
        # if len(z_values) <= 0:
        #     continue
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

    #find intersection
    # shared_min = max(rest_min_at_zmin, rest_min_at_zmax)
    # shared_max = min(rest_max_at_zmin, rest_max_at_zmax)

    # find union
    shared_min = min(rest_min_at_zmin, rest_min_at_zmax)
    shared_max = max(rest_max_at_zmin, rest_max_at_zmax)

    print(f"Shared rest-frame range: {shared_min:.1f} to {shared_max:.1f} Å")

    wavelength_grid = np.arange(np.floor(shared_min), np.ceil(shared_max), 1)

    print(f"Final wavelength grid: {len(wavelength_grid)} bins from {wavelength_grid[0]} to {wavelength_grid[-1]} Å")

    return wavelength_grid

# In[8]:
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

# In[9]:
def process_galaxy_chunk(galaxy_chunk, templates, dr_name, wavelength_grid, galaxy_type):
    """Process a chunk of galaxies and return stacking arrays"""
    bands = ['B', 'R', 'Z']

    # Initialize arrays for this chunk
    weighted_flux_stack = np.zeros_like(wavelength_grid, dtype=np.float32)
    ivar_stack = np.zeros_like(wavelength_grid, dtype=np.float32) 
    n_spectra = np.zeros_like(wavelength_grid, dtype=int)
    galaxy_count = np.zeros_like(wavelength_grid, dtype=int)

    processed_count = 0
    total_galaxies_processed = 0  
    pixels_per_fiber = []
    total_pixels_assigned = 0
    
    for galaxy in galaxy_chunk:
        total_galaxies_processed += 1
        # Apply redshift cuts
        z = galaxy['Z']
        # if galaxy_type == 'BGS' and not (0 <= z <= 0.5):
        #     continue
        # elif galaxy_type == 'LRG' and not (0.3 <= z <= 1):
        #     continue

        # Combine all bands for this galaxy
        all_wave_rest = []
        all_flux = []
        all_ivar = []

        fiber_pixel_count = 0

        for band in bands:
            # Compute residual
            residual, rest_wavelength, _ = compute_residual_spectrum(galaxy, dr_name, templates, band)
            
            ivar = galaxy[f'{band}_IVAR']
            mask = galaxy[f'{band}_MASK']
            
            # Only use good data
            valid = (mask == 0) & (ivar > 0) & np.isfinite(residual) & np.isfinite(ivar)
            in_range = (rest_wavelength >= wavelength_grid[0]) & (rest_wavelength <= wavelength_grid[-1])
            
            final_mask = valid & in_range
            
            if np.any(final_mask):
                band_pixels = np.sum(final_mask)
                fiber_pixel_count += band_pixels
                
                all_wave_rest.append(rest_wavelength[final_mask])
                all_flux.append(residual[final_mask])
                all_ivar.append(ivar[final_mask])
        
        if not all_wave_rest:
            continue

        
        # Combine all bands
        wave_rest_combined = np.concatenate(all_wave_rest)
        flux_combined = np.concatenate(all_flux)
        ivar_combined = np.concatenate(all_ivar)
        
        # Clean up intermediate arrays
        del all_wave_rest, all_flux, all_ivar
        
        if len(wave_rest_combined) == 0:
            continue
            
        total_pixels_assigned += len(wave_rest_combined)

        # Bin pixels
        bin_indices = np.searchsorted(wavelength_grid, wave_rest_combined, side='left')
        valid_bins_mask = (bin_indices >= 0) & (bin_indices < len(wavelength_grid))
        
        if not np.any(valid_bins_mask):
            continue
            
        bin_indices = bin_indices[valid_bins_mask]
        flux_combined = flux_combined[valid_bins_mask]
        ivar_combined = ivar_combined[valid_bins_mask]

        pixels_this_galaxy = len(bin_indices)
        pixels_per_fiber.append(pixels_this_galaxy)
        
        # Update stacking arrays
        weights = ivar_combined * flux_combined
        weighted_flux_stack += np.bincount(bin_indices, weights=weights, minlength=len(wavelength_grid))
        ivar_stack += np.bincount(bin_indices, weights=ivar_combined, minlength=len(wavelength_grid))
        n_spectra += np.bincount(bin_indices, minlength=len(wavelength_grid))
        
        # Track which bins this galaxy contributes to
        unique_bins = np.unique(bin_indices)
        galaxy_count[unique_bins] += 1
        
        processed_count += 1
        
        # Clean up
        del wave_rest_combined, flux_combined, ivar_combined, bin_indices, weights
    
    return weighted_flux_stack, ivar_stack, n_spectra, galaxy_count, processed_count, pixels_per_fiber, total_galaxies_processed,total_pixels_assigned
        


def stack_spectra(input_files,templates, dr_name,galaxy_type):
    print("Starting to stack spectra")
    
    #wavelength grid from residuals
    wavelength_grid = determine_wavelength_grid(input_files,dr_name,galaxy_type)
    final_galaxy_count = np.zeros_like(wavelength_grid, dtype=int)

    # Initialize arrays for stacking
    # numerator (ΣFω)
    final_weighted_flux = np.zeros_like(wavelength_grid, dtype=np.float32)
    # denominator (Σω)
    final_ivar = np.zeros_like(wavelength_grid, dtype=np.float32)
    # track spectra in each bin
    final_n_spectra = np.zeros_like(wavelength_grid, dtype=int)

    # checks
    all_pixels_per_fiber = []
    total_galaxies_attempted = 0
    total_files_processed = len(input_files)
    total_processed = 0
    total_pixels_assigned = 0

    chunk_size = 25

    for file_idx, filename in enumerate(input_files):
        print(f"\nProcessing file {file_idx+1}/{len(input_files)}: {os.path.basename(filename)}")
        print_memory(f"Before loading file {file_idx+1}")

        file_galaxies_attempted = 0
        file_processed = 0

        # Load file
        galaxy_table = Table.read(filename, path=dr_name)
        print(f"Loaded {len(galaxy_table)} galaxies")

        # Process in chunks 
        file_processed = 0
        for chunk_start in range(0, len(galaxy_table), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(galaxy_table))
            galaxy_chunk = galaxy_table[chunk_start:chunk_end]
            
            # Process this chunk
            w_flux, ivar, n_spec, gal_count, processed,pixels_per_fiber, galaxies_attempted,chunk_pixels = process_galaxy_chunk(
                galaxy_chunk, templates, dr_name, wavelength_grid, galaxy_type
            )

            # Accumulate results
            final_weighted_flux += w_flux.astype(np.float32)
            final_ivar += ivar.astype(np.float32)
            final_n_spectra += n_spec
            final_galaxy_count += gal_count
            total_pixels_assigned += chunk_pixels

            all_pixels_per_fiber.extend(pixels_per_fiber)
            file_processed += processed
            file_galaxies_attempted += galaxies_attempted

            # Clean up chunk
            del galaxy_chunk, w_flux, ivar, n_spec, gal_count

            if chunk_start % (chunk_size * 10) == 0:
                print(f"  Processed {min(chunk_end, len(galaxy_table))}/{len(galaxy_table)} galaxies")
                gc.collect()

        total_processed += file_processed
        total_galaxies_attempted += file_galaxies_attempted
        
        print(f"File summary: {file_processed} valid galaxies, {file_galaxies_attempted} total galaxies")

        del galaxy_table
        gc.collect()
        print_memory(f"After processing file {file_idx+1}")
        
    pixel_array = np.array(all_pixels_per_fiber)
    print(f"Total fibers with valid pixels: {len(pixel_array)}")
    print(f"Average pixels per fiber: {np.mean(pixel_array):.1f}")
    print(f"Min pixels per fiber: {np.min(pixel_array)}")
    print(f"Max pixels per fiber: {np.max(pixel_array)}")

    print(f"Total files processed: {total_files_processed}")
    print(f"Total galaxies attempted: {total_galaxies_attempted}")
    print(f"Total galaxies successfully processed: {total_processed}")
    
    # Final stacked spectrum
    valid_bins = (final_ivar > 0)
    stacked_flux = np.zeros_like(wavelength_grid, dtype=float)
    stacked_error = np.zeros_like(wavelength_grid, dtype=float)
    
    # Final equation
    stacked_flux[valid_bins] = final_weighted_flux[valid_bins] / final_ivar[valid_bins]
    # Error: 1/√ Σω
    stacked_error[valid_bins] = np.sqrt(1.0 / final_ivar[valid_bins])

    print(f"Processed {total_processed} galaxies")
    print(f"Max pixels/bin: {final_n_spectra.max()}, Min pixels/bin (valid): {final_n_spectra[valid_bins].min()}")
    print(f"Valid bins: {np.sum(valid_bins)}/{len(valid_bins)}")
    if np.sum(valid_bins) > 0:
        print(f"Flux range: {np.min(stacked_flux[valid_bins])} to {np.max(stacked_flux[valid_bins])}")
        print(f"Wavelength range (valid): {wavelength_grid[valid_bins].min()} - {wavelength_grid[valid_bins].max()} Å")
    else:
        print("WARNING: No valid bins in final stack!")

    all_galaxy_mask = (final_galaxy_count == total_processed)
    print(f"Bins with all galaxies: {np.sum(all_galaxy_mask)}")
    print(f"Average number of pixels across all spectra: {total_pixels_assigned / total_processed}")


    return wavelength_grid, stacked_flux, stacked_error, final_n_spectra, valid_bins, all_galaxy_mask, total_processed

# In[10]:

def main(dr_name,galaxy_type):
    print_memory("Start")
    start_time = time.time()  

    input_dir = f"/pscratch/sd/a/aizaa/{galaxy_type}/galaxy_{dr_name}"
    input_files = sorted(glob.glob(os.path.join(input_dir, f'{galaxy_type}_galaxy_*.h5')))
    print(f"Found {len(input_files)} files to combine")

    base_output_dir = f"/pscratch/sd/a/aizaa/results" 
    plot_dir = f"/pscratch/sd/a/aizaa/results/{galaxy_type}/{dr_name}_galaxy"

    print('loading templates')
    templates = load_templates(dr_name)
    print_memory("After loading templates")

    print_memory("Before stacking")
    wave, flux, error, n_spectra, valid_bins, all_galaxy_mask,processed_galaxies = stack_spectra(input_files,templates, dr_name,galaxy_type)
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
                n_spectra, continuum, noise, base_output_dir, galaxy_type,dr_name,processed_galaxies)

    print("Done with calculations!")
    print_memory("done with filters")
    # smoothed curves alone:
    smoothed_narrow = snr_narrow * noise
    smoothed_broad = snr_broad * noise

    plot(wave,flux,continuum,residual,smoothed_narrow, smoothed_broad,noise,snr_narrow,snr_broad,valid_bins,all_galaxy_mask,plot_dir)
    print_memory("after plotting")
    print("Done plotting!")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal job time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


# In[ ]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DESI sky galaxys')
    parser.add_argument('dr_name')
    parser.add_argument('galaxy_type')
    args = parser.parse_args()
    main(dr_name=args.dr_name,galaxy_type=args.galaxy_type)
    

# In[ ]: