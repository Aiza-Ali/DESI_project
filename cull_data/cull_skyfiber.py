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
from astropy.coordinates import SkyCoord
from astropy import units as u
import gc
import psutil
from collections import defaultdict

# In[2]:

def get_spectral_data(coadd_file, bands= ['B','R','Z']):
    """returns table that will contain wavelength, flux, resolution, ivar, mask for targetids in each band"""
    
    with fits.open(coadd_file, memmap=True, lazy_load_hdus=True) as hdul:
            
        # Get TARGETID first
        targetids = hdul['FIBERMAP'].data['TARGETID']
        n_target = len(targetids)

        # initialize result dictionary
        result = {'TARGETID': targetids}
            
        for band in bands: 
            wave_data = hdul[f'{band}_WAVELENGTH'].data
            flux_data = hdul[f'{band}_FLUX'].data
            resolution_data = hdul[f'{band}_RESOLUTION'].data
            ivar_data = hdul[f'{band}_IVAR'].data
            mask_data = hdul[f'{band}_MASK'].data
                
            # Use list comprehension 
            result[f'{band}_WAVELENGTH'] = [wave_data] * n_target
            result[f'{band}_FLUX'] = [flux_data[i] for i in range(n_target)]
            result[f'{band}_RESOLUTION'] = [resolution_data[i] for i in range(n_target)]
            result[f'{band}_IVAR'] = [ivar_data[i] for i in range(n_target)]
            result[f'{band}_MASK'] = [mask_data[i] for i in range(n_target)]

            del wave_data, flux_data, resolution_data, ivar_data, mask_data
    
    gc.collect()
    return Table(result)

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


# In[3]:

def filters(galaxy_type):
    """return filter dictionaries for sky fibers and galaxy"""

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

    def objtype_filter(x):
        return x.astype(str) == "SKY"

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

    sky_filters = {
        "OBJTYPE": objtype_filter
    }
    return galaxy_filters, sky_filters

# In[4]:

def apply_filters(table, filters):
    """Apply filters """
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


# In[5]:


def find_pairs(sky_coords, gal_coords, sky_z, gal_z):
    """Process sky-galaxy pairs in chunks, return pairs of skyfibers/galaxies
        NOTE: skyfibers can repeat"""
    chunk_size=500
    n_sky = len(sky_coords)
    n_gal = len(gal_coords)

    # list of valid pairs
    valid_pairs = []
    
    # Process sky fibers in chunks
    for sky_start in range(0, n_sky, chunk_size):
        sky_end = min(sky_start + chunk_size, n_sky)
        sky_chunk = sky_coords[sky_start:sky_end]
        sky_z_chunk = sky_z[sky_start:sky_end]
        
        # for each sky chunk, Process galaxies in chunks
        for gal_start in range(0, n_gal, chunk_size):
            gal_end = min(gal_start + chunk_size, n_gal)
            gal_chunk = gal_coords[gal_start:gal_end]
            gal_z_chunk = gal_z[gal_start:gal_end]
            
            # Calculate separation between galaxy/skyfiber for this chunk
            sep_matrix = sky_chunk[:, None].separation(gal_chunk[None, :]).arcsecond
            # calculate redshift sep between galaxy/skyfiber for this chunk
            z_diff_matrix = np.abs(sky_z_chunk[:, None] - gal_z_chunk[None, :])
            z_sep_matrix = z_diff_matrix / (1 + gal_z_chunk[None, :])
            
            
            # check why BGS has no pairs within 2.5''
            ang_sep_leq_3 = np.count_nonzero(sep_matrix <= 3)
            if ang_sep_leq_3 > 0:
                print(f"there is {ang_sep_leq_3} pair(s) with angular separation less than 3!")
                bad_z_ang_sep_leq_3 = np.count_nonzero((sep_matrix <= 3) & (z_sep_matrix <= 0.05))
                print(f" {bad_z_ang_sep_leq_3} of those pair(s) have bad redshift separation")

            # Find valid pairs (sep/redshift)
            sky_indices, gal_indices = np.where((sep_matrix <= 20) & (z_sep_matrix > 0.05))
            # convert to global indices
            for local_sky_idx, local_gal_idx in zip(sky_indices, gal_indices):
                global_sky_idx = sky_start + local_sky_idx
                global_gal_idx = gal_start + local_gal_idx
                valid_pairs.append((global_sky_idx, global_gal_idx))


            del sep_matrix, z_diff_matrix, z_sep_matrix
            del sky_indices, gal_indices
            del gal_chunk, gal_z_chunk
        del sky_chunk, sky_z_chunk
    
    return valid_pairs


# In[6]:


def combine_fibermap_redrock(bright_files, dark_files):
    """for same Healpix, combine bright and dark data"""
    combined_fibermap = []
    combined_redrock = []

    for coadd_file in bright_files:
        # corresponding redrock file
        redrock_file = coadd_file.replace("coadd-", "redrock-")
    
        # open coadd/redrock file
        with fits.open(coadd_file, memmap=True) as hdul, fits.open(redrock_file, memmap=True) as rhdul:
            # load necessary columns and data
            fibermap_data = hdul['FIBERMAP'].data
            redrock_data = rhdul['REDSHIFTS'].data

            fibermap_one = Table()
            fibermap_one['TARGETID'] = fibermap_data['TARGETID']
            fibermap_one['TARGET_RA'] = fibermap_data['TARGET_RA']
            fibermap_one['TARGET_DEC'] = fibermap_data['TARGET_DEC']
            fibermap_one['BGS_TARGET'] = fibermap_data['BGS_TARGET']
            fibermap_one['DESI_TARGET'] = fibermap_data['DESI_TARGET']
            fibermap_one['OBJTYPE'] = fibermap_data['OBJTYPE']
        
            redrock_one = Table()
            redrock_one['TARGETID'] = redrock_data['TARGETID']
            redrock_one['Z'] = redrock_data['Z']
            redrock_one['ZWARN'] = redrock_data['ZWARN']
            redrock_one['SPECTYPE'] = redrock_data['SPECTYPE']
            redrock_one['DELTACHI2'] = redrock_data['DELTACHI2']
            redrock_one['COEFF'] = [np.array(i) for i in redrock_data['COEFF']]

            combined_fibermap.append(fibermap_one)
            combined_redrock.append(redrock_one)

            del fibermap_data, redrock_data
            del fibermap_one, redrock_one

    for coadd_file in dark_files:
        # corresponding redrock file
        redrock_file = coadd_file.replace("coadd-", "redrock-")
    
        # open coadd/redrock file
        with fits.open(coadd_file, memmap=True) as hdul, fits.open(redrock_file, memmap=True) as rhdul:
            # load necessary columns and data
            fibermap_data = hdul['FIBERMAP'].data
            redrock_data = rhdul['REDSHIFTS'].data

            fibermap_one = Table()
            fibermap_one['TARGETID'] = fibermap_data['TARGETID']
            fibermap_one['TARGET_RA'] = fibermap_data['TARGET_RA']
            fibermap_one['TARGET_DEC'] = fibermap_data['TARGET_DEC']
            fibermap_one['BGS_TARGET'] = fibermap_data['BGS_TARGET']
            fibermap_one['DESI_TARGET'] = fibermap_data['DESI_TARGET']
            fibermap_one['OBJTYPE'] = fibermap_data['OBJTYPE']
        
            redrock_one = Table()
            redrock_one['TARGETID'] = redrock_data['TARGETID']
            redrock_one['Z'] = redrock_data['Z']
            redrock_one['ZWARN'] = redrock_data['ZWARN']
            redrock_one['SPECTYPE'] = redrock_data['SPECTYPE']
            redrock_one['DELTACHI2'] = redrock_data['DELTACHI2']
            redrock_one['COEFF'] = [np.array(i) for i in redrock_data['COEFF']]

            combined_fibermap.append(fibermap_one)
            combined_redrock.append(redrock_one)

            del fibermap_data, redrock_data
            del fibermap_one, redrock_one
        gc.collect()

    if combined_fibermap and combined_redrock:
        stacked_fibermap = ap_vstack(combined_fibermap)
        stacked_redrock = ap_vstack(combined_redrock)
        del combined_fibermap, combined_redrock
        gc.collect()

        combined_one = ap_join(stacked_fibermap, stacked_redrock, keys=['TARGETID'])

        del stacked_fibermap, stacked_redrock
        return combined_one
    else:
        print("no fibermap or redrock data to combine!")
        return None

def process_healpix_group(bright_files,dark_files,galaxy_filters,sky_filters):
        """process group of file with same pixel #"""

        combined_one = combine_fibermap_redrock(bright_files,dark_files)
        if not combined_one:
            print("no data in this healpix group")
            return None
        
        # apply filters
        galaxies = apply_filters(combined_one, galaxy_filters)
        sky_fibers = apply_filters(combined_one, sky_filters)

        del combined_one
        gc.collect()

        if len(galaxies) == 0 or len(sky_fibers) == 0:
            print(f'no good galaxy and/or skyfiber in this healpix group')
            del galaxies, sky_fibers
            gc.collect()
            return None

        # get spectral data
        all_spectral_data = []

        # bright files
        for file in bright_files:
            spectral_data = get_spectral_data(file)
            if spectral_data:
                all_spectral_data.append(spectral_data)
                del spectral_data
                gc.collect()
            else:
                print("no spectral data found in bright {file}")

        # dark files
        for file in dark_files:
            spectral_data = get_spectral_data(file)
            if spectral_data:
                all_spectral_data.append(spectral_data)
                del spectral_data
                gc.collect()
            else:
                print("no spectral data found in dark {file}")

        # combine spectral data
        combined_spectral_data = ap_vstack(all_spectral_data)
        del all_spectral_data
        gc.collect()


        # join filtered data
        galaxies = ap_join(galaxies, combined_spectral_data, keys=['TARGETID'])
        sky_fibers = ap_join(sky_fibers, combined_spectral_data, keys=['TARGETID'])

        del combined_spectral_data
        gc.collect()

        # get coordinates
        gal_coords = SkyCoord(ra=galaxies['TARGET_RA']*u.deg, dec=galaxies['TARGET_DEC']*u.deg)
        sky_coords = SkyCoord(ra=sky_fibers['TARGET_RA']*u.deg, dec=sky_fibers['TARGET_DEC']*u.deg)
            
        # Calculate separations more efficiently
        valid_pairs = find_pairs(sky_coords, gal_coords, sky_fibers['Z'], galaxies['Z'])
        del gal_coords, sky_coords
        gc.collect()
            
        # Find matches more efficiently, use skyfiber only once
        good_sky_fibers = []
            
        if valid_pairs:
            # track skyfibers already used
            sky_indices_used = set()
            
            for sky_idx, gal_idx in valid_pairs:
                # Only use each sky fiber once (first match)
                if sky_idx not in sky_indices_used:
                    sky_indices_used.add(sky_idx)
                    
                    sky = sky_fibers[sky_idx]
                    gal = galaxies[gal_idx]
                        
                    # Build match dictionary 
                    match = {
                        'TARGETID': sky['TARGETID'],
                        'Z': sky['Z'],
                        'COEFF': sky['COEFF'],
                        'TARGET_RA': sky['TARGET_RA'],
                        'TARGET_DEC': sky['TARGET_DEC'],
                        'NEAR_GALAXY_ID': gal['TARGETID'],
                        'NEAR_GALAXY_Z': gal['Z'],
                        'NEAR_GALAXY_RA': gal['TARGET_RA'],
                        'NEAR_GALAXY_DEC': gal['TARGET_DEC']
                       }
                        
                    # Add spectral data
                    for band in ['B','R','Z']:
                        for col in ['WAVELENGTH', 'RESOLUTION', 'FLUX', 'IVAR', 'MASK']:
                            match[f'{band}_{col}'] = sky[f'{band}_{col}']
                        
                    good_sky_fibers.append(match)
            
            # Clean up memory
            del sky_indices_used, valid_pairs
            gc.collect()
            if good_sky_fibers:
                return good_sky_fibers
            else:
                print('no good skyfibers found :-( )')
                return

# In[7]:
    
def save_results(results, job_id, outfile,dr_name):
    """ save results to file  """
    if not results:
        return None

    # write table to file
    result_table = Table(results)
    result_table.write(outfile, path=dr_name, format='hdf5', overwrite=True)
    print(f"\nSaved {len(results)} sky fibers to {outfile}")
    del result_table
    gc.collect()
    return outfile
                

# In[8]:


def process_file_subset(bright_healpix,dark_healpix, 
                        galaxy_filters, sky_filters,
                        start_idx, end_idx, 
                        job_id, outdir,dr_name,galaxy_type):
    """ process a subset of healpix groups """
    subset_start_time = time.time()
    print_memory_usage()

    # healpix list and subset
    all_healpix = set(bright_healpix.keys()).union(dark_healpix.keys())
    healpix_list = sorted(list(all_healpix))
    healpix_subset = healpix_list[start_idx:end_idx]

    print(f"Job {job_id}: Processing HEALPix groups {start_idx} to {end_idx-1} ({len(healpix_subset)} groups)")

    # Create file path
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{galaxy_type}_skyfibers_{job_id}.h5")

    # Accumulate ALL results for this job
    all_results = []  
    total_found = 0

    # process groups sequentially
    for i, healpix in enumerate(healpix_subset):
        if i % 10 == 0:
            print(f"Job {job_id}: Processing group {i+1}/{len(healpix_subset)} (pixel {healpix})")
            print_memory_usage()
        
        # all files for this healpix
        bright_subset = bright_healpix.get(healpix, [])
        dark_subset = dark_healpix.get(healpix, [])
            
        # process group
        result = process_healpix_group(bright_subset,dark_subset, galaxy_filters, sky_filters)
        if result:
            all_results.extend(result)
            total_found += len(result)
            del result
            
            if total_found % 100 == 0:
                print(f"Job {job_id}: Total skyfibers found so far: {total_found}")
        
        #  garbage collection every few files
        if i % 2 == 0:
            gc.collect()

    save_start_time = time.time()
    # Save ALL results for this job at the end
    if all_results:
        print(f"Job {job_id}: Saving final results of {len(all_results)} sky fibers")
        save_results(all_results, job_id, outfile,dr_name)
        del all_results
        force_garbage_collect()

    save_end_time = time.time()
    subset_end_time = time.time()
    total_subset_time = subset_end_time - subset_start_time
    save_time = save_end_time - save_start_time
    processing_time = total_subset_time - save_time
        
    print(f"Job {job_id}: Completed processing. Found {total_found} sky fibers")

    # Create a status file to indicate this job completed successfully
    status_file = os.path.join(outdir, f'job_{job_id}.completed')
    with open(status_file, 'w') as f:
        f.write(f"Job {job_id} completed successfully\n")
        f.write(f"Found {total_found} sky fibers\n")
        f.write(f"Files processed: {len(healpix_subset)}\n")
        f.write(f"Start time: {datetime.fromtimestamp(subset_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.fromtimestamp(subset_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processing time: {processing_time/60:.2f} minutes\n")
        f.write(f"Save time: {save_time:.2f} seconds\n")
        f.write(f"Total time: {total_subset_time/60:.2f} minutes\n")
    
    return total_found

# In[9]:


def main(dr_name, galaxy_type):
    # Necessary to use dvs_ro read-in which is faster
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    # parameters
    start_time = time.time()
    
    # Get filters
    galaxy_filters, sky_filters = filters(galaxy_type)
    
    # Get job parameters from SLURM environment
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
    total_jobs = int(os.getenv('SLURM_ARRAY_TASK_COUNT', '1'))
    print(f"Job ID: {job_id}, Total jobs: {total_jobs}")

    # output directory
    outdir = f"/pscratch/sd/a/aizaa/{galaxy_type}/skyfiber_{dr_name}"

    # Healpix directory containing coadd files
    coadd_dir_base = f"/dvs_ro/cfs/cdirs/desi/spectro/redux/{dr_name}/healpix/main/"

    #bright/dark coadd files
    coadd_file_lst_bright = glob.glob(f"{coadd_dir_base}bright/*/*/coadd-main-bright-*.fits")
    coadd_file_lst_dark = glob.glob(f"{coadd_dir_base}dark/*/*/coadd-main-dark-*.fits")


    print(f"Found {len(coadd_file_lst_bright)} bright files with recursive glob.")
    print(f"Found {len(coadd_file_lst_dark)} dark files with recursive glob.")

    # group files by healpix number
    bright_healpix,dark_healpix = group_files_by_healpix(coadd_file_lst_bright,coadd_file_lst_dark)
    all_healpix = set(bright_healpix).union(dark_healpix)
    total_groups = len(all_healpix)

    print(f"Found {total_groups} unique HEALPix groups to process")

    # Calculate group range for this job
    groups_per_job = total_groups // total_jobs
    remainder = total_groups % total_jobs
    
    # Distribute remainder files among first jobs
    if job_id < remainder:
        start_idx = job_id * (groups_per_job + 1)
        end_idx = start_idx + groups_per_job + 1
    else:
        start_idx = remainder * (groups_per_job + 1) + (job_id - remainder) * groups_per_job
        end_idx = start_idx + groups_per_job

    # process data
    print('Starting compilation')
    # Process assigned files
    n_found = process_file_subset(bright_healpix, dark_healpix,
                                  galaxy_filters, sky_filters, 
                                  start_idx, end_idx, job_id, outdir, dr_name, galaxy_type)
    
    print(f"Job {job_id} completed in {(time.time()-start_time)/60:.2f} minutes")
    print(f"Job {job_id} found {n_found} sky fibers")


# In[10]:
def group_files_by_healpix(bright_list,dark_list):
    # initialize dictionaries
    bright_healpix_dict = defaultdict(list)
    dark_healpix_dict = defaultdict(list)
    # group bright
    for file in bright_list:
        healpix = get_healpix_number(file)
        if healpix is not None:
            bright_healpix_dict[healpix].append(file)

    # group dark
    for file in dark_list:
        healpix = get_healpix_number(file)
        if healpix is not None:
            dark_healpix_dict[healpix].append(file)

    return bright_healpix_dict, dark_healpix_dict



def get_healpix_number(filename):
    parts = filename.split('/')
    return(parts[-1])

def combine_spectral_data(bright_table,dark_table):
    if bright_table is None:
        return dark_table
    elif dark_table is None:
        return bright_table
    else:
        return ap_vstack([bright_table,dark_table])


if __name__ == "__main__":
    # parser obj to handle command line arguments
    parser = argparse.ArgumentParser(description='Process DESI galaxies')
    # required arguments in order
    parser.add_argument('dr_name', help='Data release name (e.g., iron)')
    parser.add_argument('galaxy_type', help='Galaxy type (e.g., BGS)')
    args = parser.parse_args()
    
    main(dr_name=args.dr_name,galaxy_type = args.galaxy_type)