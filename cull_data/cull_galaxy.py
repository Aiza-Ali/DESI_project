#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import glob
import time
from datetime import datetime
import argparse
import os
from astropy.table import Table, MaskedColumn
from astropy.table import join as ap_join
from astropy.table import vstack as ap_vstack
from astropy.io import fits
import gc
import psutil

# In[2]:

def get_spectral_data(coadd_file,bands= ['B','R','Z']):
    """returns table that will contain wavelength, flux, resolution, ivar, mask for each band"""
    # open each coadd file
    with fits.open(coadd_file) as hdul:
        # get TARGETID in results
        targetids = hdul['FIBERMAP'].data['TARGETID']
        n_target = len(targetids)
        result = {'TARGETID': targetids}
        
        for band in bands:
            # get wavelength (1d array) (as column in table)
            wave_data = hdul[f'{band}_WAVELENGTH'].data
            # get flux (2d array) (another column in table)
            flux_data = hdul[f'{band}_FLUX'].data
            # get resolution (3d array) (another column in table)
            resolution_data = hdul[f'{band}_RESOLUTION'].data
            # get ivar 2d
            ivar_data = hdul[f'{band}_IVAR'].data
            # get mask 2d
            mask_data = hdul[f'{band}_MASK'].data

            # use list comprehension
            result[f'{band}_WAVELENGTH'] = [wave_data] * n_target
            result[f'{band}_FLUX'] = [flux_data[i] for i in range(n_target)]
            result[f'{band}_RESOLUTION'] = [resolution_data[i] for i in range(n_target)]
            result[f'{band}_IVAR'] = [ivar_data[i] for i in range(n_target)]
            result[f'{band}_MASK'] = [mask_data[i] for i in range(n_target)]

            del wave_data, flux_data, resolution_data, ivar_data, mask_data
            gc.collect()
            
    return Table(result)


# In[3]:

def filters(galaxy_type):
    """return filter dictionaries for galaxy"""

    def zwarn_filter(x):
        return x == 0

    def deltachi2_filter(x):
        return x > 40.0

    def spectype_filter(x):
        return x.astype(str) == "GALAXY"
    
    def bgs_target_filter(x):
        return x != 0
    
    def lrg_target_filter(x):
        return (x & 1) == 1

    if galaxy_type == 'BGS':
        galaxy_filters = {
            "ZWARN": zwarn_filter,
            "DELTACHI2": deltachi2_filter,
            "SPECTYPE": spectype_filter,
            "BGS_TARGET": bgs_target_filter
            }
    else:
        galaxy_filters = {
        "ZWARN": zwarn_filter,
        "DELTACHI2": deltachi2_filter,
        "SPECTYPE": spectype_filter,
        "DESI_TARGET": lrg_target_filter
        }
        
    return galaxy_filters

def apply_filters(table, filters):
    """Apply filters using vectorized operations"""
    if len(table) == 0:
        return table
    # all true
    mask = np.ones(len(table), dtype=bool)
    # loop through filters
    for col, condition in filters.items():
        if col in table.colnames:
            mask &= condition(table[col])
    # return filtered table
    return table[mask]

# In[4]:

def process_file(coadd_file, galaxy_filters, retain_columns):
    """Process a single file """
    # corresponding redrock file
    redrock_file = coadd_file.replace("coadd-", "redrock-")

    # open coadd/redrock file
    with fits.open(coadd_file) as hdul, fits.open(redrock_file) as rhdul:
        # load necessary columns
        fibermap_data = Table(hdul['FIBERMAP'].data)
        redrock_data = Table(rhdul['REDSHIFTS'].data)

        # get NEEDED data
        fibermap_one = Table()
        fibermap_one['TARGETID'] = fibermap_data['TARGETID']
        fibermap_one['TARGET_RA'] = fibermap_data['TARGET_RA']
        fibermap_one['TARGET_DEC'] = fibermap_data['TARGET_DEC']
        fibermap_one['BGS_TARGET'] = fibermap_data['BGS_TARGET']
        fibermap_one['OBJTYPE'] = fibermap_data['OBJTYPE']
        fibermap_one['DESI_TARGET'] = fibermap_data['DESI_TARGET']


        del fibermap_data
        gc.collect()

        redrock_one = Table()
        redrock_one['TARGETID'] = redrock_data['TARGETID']
        redrock_one['Z'] = redrock_data['Z']
        redrock_one['ZWARN'] = redrock_data['ZWARN']
        redrock_one['SPECTYPE'] = redrock_data['SPECTYPE']
        redrock_one['DELTACHI2'] = redrock_data['DELTACHI2']
        redrock_one['COEFF'] = [np.array(i) for i in redrock_data['COEFF']]

        del redrock_data
        gc.collect()

        # join tables
        combined_one = ap_join(fibermap_one, redrock_one, keys=['TARGETID'])

        del fibermap_one, redrock_one
        gc.collect()

        # apply filters
        galaxies = apply_filters(combined_one, galaxy_filters)

        del combined_one
        gc.collect()

        if len(galaxies) == 0:
            print(f'no galaxy passes all filters in file: {coadd_file}')
            del galaxies
            return None

        # get the other data
        wave_flux_tab = get_spectral_data(coadd_file, bands = ['B','R','Z'])

        # join filtered data
        galaxies = ap_join(galaxies, wave_flux_tab, keys=['TARGETID'])

        del wave_flux_tab
        gc.collect()

        # list of good galaxies
        good_galaxies = []

        if len(galaxies) > 0:
            for gal in galaxies:
                # Build match dictionary 
                match = {
                    'TARGETID': gal['TARGETID'],
                    'Z': gal['Z'],
                    'COEFF': gal['COEFF']
                    }
                        
                # Add spectral data
                for band in ['B','R','Z']:
                    for col in ['WAVELENGTH', 'RESOLUTION', 'FLUX', 'IVAR', 'MASK']:
                        match[f'{band}_{col}'] = gal[f'{band}_{col}']
                        
                good_galaxies.append(match)
            
            # Clean up memory
            del galaxies
            gc.collect()
            
            return good_galaxies if good_galaxies else None


# In[5]:

def save_results(results, job_id, outfile,dr_name):
    """ save results to file  """
    if not results:
        return None

    # write table to file
    intermediate_file = outfile
    result_table = Table(results)
    result_table.write(intermediate_file, path=dr_name, format='hdf5', overwrite=True)
    print(f"\nSaved {len(results)} galaxies to {intermediate_file}")

    del result_table
    gc.collect()
    return intermediate_file


# In[6]:

def main(dr_name,galaxy_type):
    # Necessary to use dvs_ro read-in which is faster
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    # parameters
    start_time = time.time()
    # Get filters
    galaxy_filters = filters(galaxy_type)
    # Get job parameters from SLURM environment
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
    total_jobs = int(os.getenv('SLURM_ARRAY_TASK_COUNT', '1'))
    print(f"Job ID: {job_id}, Total jobs: {total_jobs}")
    
    # output directory
    outdir = f"/pscratch/sd/a/aizaa/{galaxy_type}/galaxy_{dr_name}"

    # Directory containing coadd files
    coadd_dir = f"/dvs_ro/cfs/cdirs/desi/spectro/redux/{dr_name}/tiles/cumulative/"

    #all coadd files
    coadd_file_lst = glob.glob(f"{coadd_dir}/*/*/coadd-*.fits")

    print(f"Found {len(coadd_file_lst)} files with recursive glob.")

    # Calculate file range for this job
    files_per_job = len(coadd_file_lst) // total_jobs
    remainder = len(coadd_file_lst) % total_jobs
    
    # Distribute remainder files among first few jobs
    if job_id < remainder:
        start_idx = job_id * (files_per_job + 1)
        end_idx = start_idx + files_per_job + 1
    else:
        start_idx = remainder * (files_per_job + 1) + (job_id - remainder) * files_per_job
        end_idx = start_idx + files_per_job

    retain_columns = ['TARGETID', 'Z', 'COEFF', 'TARGET_RA', 'TARGET_DEC', 
                  'B_WAVELENGTH', 'B_FLUX','B_RESOLUTION', 'B_IVAR', 'B_MASK',
                  'R_WAVELENGTH', 'R_FLUX','R_RESOLUTION', 'R_IVAR', 'R_MASK',
                  'Z_WAVELENGTH', 'Z_FLUX', 'Z_RESOLUTION', 'Z_IVAR', 'Z_MASK',]
    
    # process data
    print('Starting compilation')
    # Process assigned files
    n_found = process_file_subset(coadd_file_lst, galaxy_filters, retain_columns, start_idx, end_idx, job_id, outdir, dr_name,galaxy_type, save_every=5)
    
    print(f"Job {job_id} completed in {(time.time()-start_time)/60:.2f} minutes")
    print(f"Job {job_id} found {n_found} galaxies")

# In[7]:

def process_file_subset(coadd_file_lst, galaxy_filters, retain_columns, start_idx, end_idx, job_id, outdir,dr_name, galaxy_type, save_every=5):
    """ process a subset of files """
    subset_start_time = time.time()
    print_memory_usage()

    # get subset of files
    file_subset = coadd_file_lst[start_idx:end_idx]
    print(f"Job {job_id}: Processing files {start_idx} to {end_idx-1} ({len(file_subset)} files)")

    # Create file path
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{galaxy_type}_galaxy_{job_id}.h5")

    # Accumulate ALL results for this job
    all_results = []  
    total_found = 0

    # process files sequentially
    for i, coadd_file in enumerate(file_subset):
        if i % 10 == 0:
            print(f"Job {job_id}: Processing file {i+1}/{len(file_subset)}")
            print_memory_usage()
            
        # process file
        result = process_file(coadd_file, galaxy_filters, retain_columns)
        if result:
            all_results.extend(result)
            total_found += len(result)
            
            if total_found % 100 == 0:
                print(f"Job {job_id}: Total galaxies found so far: {total_found}")
        
        #  garbage collection every few files
        if i % 2 == 0:
            gc.collect()
            
    # Save ALL results for this job at the end
    save_start_time = time.time()
    if all_results:
        print(f"Job {job_id}: Saving final results of {len(all_results)} galaxies")
        save_results(all_results, job_id, outfile,dr_name)
        force_garbage_collect()
    save_end_time = time.time()

    subset_end_time = time.time()
    total_subset_time = subset_end_time - subset_start_time
    save_time = save_end_time - save_start_time
    processing_time = total_subset_time - save_time
    print(f"Job {job_id}: Completed processing. Found {total_found} galaxies")

    # Create a status file to indicate this job completed successfully
    status_file = os.path.join(outdir, f'job_{job_id}.completed')
    with open(status_file, 'w') as f:
        f.write(f"Job {job_id} completed successfully\n")
        f.write(f"Found {total_found} galaxies\n")
        f.write(f"Files processed: {len(file_subset)}\n")
        f.write(f"Start time: {datetime.fromtimestamp(subset_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.fromtimestamp(subset_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processing time: {processing_time/60:.2f} minutes\n")
        f.write(f"Save time: {save_time:.2f} seconds\n")
        f.write(f"Total time: {total_subset_time/60:.2f} minutes\n")
    
    return total_found


# In[8]:

def force_garbage_collect():
    """Force garbage collection and print memory usage"""
    collected = gc.collect()
    print(f"Garbage collected {collected} objects")
    print_memory_usage()

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")


# In[ ]:

if __name__ == "__main__":

    # parser obj to handle command line arguments
    parser = argparse.ArgumentParser(description='Process DESI galaxies')
    # required arguments in order
    parser.add_argument('dr_name', help='Data release name (e.g., iron)')
    parser.add_argument('galaxy_type', help='Galaxy type (e.g., BGS)')
    args = parser.parse_args()
    
    main(dr_name=args.dr_name,galaxy_type = args.galaxy_type)