# useful objects and functions for ASL1+2 analysis
# subject dictionary
# sign flip permutation test
# tools for plotting results on the fsaverage7 surface

import os
import numpy as np
import copy
import pickle
from pathlib import Path
from itertools import combinations
import pandas as pd

import pingouin as pg
import numbers
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
import time
from collections import OrderedDict
import statsmodels.api as sm

import nilearn.datasets as datasets
from nilearn import surface
import nibabel

import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf
from PIL import Image, ImageDraw, ImageFont

## SUB DICTS

asl1_subs = {
     '001':{'group':'ASL'},'002':{'group':'RUSS'},'003':{'group':'ASL'},'004':{'group':'ASL'},
     '005':{'group':'RUSS'},'006':{'group':'RUSS'},'007':{'group':'RUSS'},'008':{'group':'RUSS'},
     '009':{'group':'ASL'},'010':{'group':'ASL'},'011':{'group':'RUSS'},'012':{'group':'ASL'},
     '013':{'group':'RUSS'},'014':{'group':'RUSS'},'015':{'group':'ASL'},'016':{'group':'ASL'},
     '017':{'group':'ASL'},'018':{'group':'ASL'},'019':{'group':'RUSS'},'020':{'group':'RUSS'}}


asl2_subs = {'001':{'group':'1'},'003':{'group':'2'},'004':{'group':'1'},
     '005':{'group':'1'},'006':{'group':'1'},'007':{'group':'1'},'008':{'group':'1'},
     '009':{'group':'1'},'010':{'group':'1'},'011':{'group':'1'},'012':{'group':'1'},
     '013':{'group':'2'},'014':{'group':'1'},'015':{'group':'2'},'016':{'group':'2'},
     '017':{'group':'2'},'018':{'group':'2'},'019':{'group':'2'},'020':{'group':'2'},
     '021':{'group':'2'},'022':{'group':'2'},'023':{'group':'2'},'024':{'group':'2'},
     '025':{'group':'2'},'026':{'group':'2'},'027':{'group':'2'},'028':{'group':'1'},
     '029':{'group':'2'},'030':{'group':'2'},'031':{'group':'1'},'032':{'group':'2'},
     '033':{'group':'1'},'034':{'group':'1'},'035':{'group':'1'},'036':{'group':'1'},
     '037':{'group':'1'},'038':{'group':'1'},'040':{'group':'1'},'041':{'group':'2'},
     '042':{'group':'2'}}


### RSA FUNCTIONS

# FUNCTIONS by D.K., A.C., and J.C. originally developed for Statics
# Modified by M.H.
#  Computes and return a null distribution (perms_norm) using target DSM as a seed, 
#  also return normed target DSM
#  **Be sure DSM is in vector form (i.e. not square)

def norm_and_perms_norm(target_dsm, n_perms = 1000):
    
#     # generate n_perms permutations of target DSM
    perm = list(range(len(target_dsm)))
    np.random.shuffle(perm)
    perms = np.zeros((n_perms,len(target_dsm)))
    print('all perms:', perms.shape)

    dsm_fake = copy.deepcopy(target_dsm)
    for i in range(n_perms):
        np.random.shuffle(dsm_fake)
        perms[i, :] = dsm_fake    
    
    # average permuted distributions, compute normed null distribution
    mu = np.mean(perms, 1)
    perms_cent = (perms.T - mu).T
    norms = np.linalg.norm(perms_cent, axis = 1)
    perms_norm = (perms_cent.T / norms).T
    print('perms_norm:', perms_norm.shape)
    
    
    # compute normed target DSM
    targ_norm = copy.deepcopy(target_dsm)
    targ_norm = targ_norm - np.mean(targ_norm)
    targ_norm = targ_norm / np.linalg.norm(targ_norm)
    print('targ_norm', targ_norm.shape)
    
    return targ_norm, perms_norm

### based on Andy's original compute_permute function ###
# for wholebrain DSMs for subject s:
#   - mean-center and norm (i.e., z-score) the DSMs
#   - compare to null model (perms_norm) using dot products and norms to compute correlation (see reference)
#   - compare to target model using above method
#   - subtract null comparison from target comparison and divide by null stdev to get a zmap

# returns both the targ_dot which is the dot product of the target with the data (the regular RSA)
#  as well as the zmap which is the targ_dot compared to the permuted null distribution with a z-test mean comparison

# correlation vs. dot product reference: https://www.quora.com/Is-there-any-relation-between-correlation-of-two-signals-and-dot-product-of-two-vectors

def normed_RSA(data, targ_norm, perms_norm):
    
    # import and norm subject's DSMs
    data = data - np.mean(data, 0)
    data = data / np.linalg.norm(data, axis=0)
    
#     # compute null comparison
    null_dot = np.dot(perms_norm, data)
    null_means = np.mean(null_dot, 0)
    null_stdevs = np.std(null_dot, 0)
    
    # compute target comparison
    targ_dot = np.dot(targ_norm, data)
    
   
    # return zmap of target comparison normalized by null comparison
    zmap = (targ_dot - null_means)/null_stdevs
    
    return targ_dot, zmap

    



### SIGN FLIP PERMUTATION TEST (PARCELS)
## Borrowed from A. Dunn

def one_sample_perm_test_signflip(
    corrs: pd.DataFrame,
    n_perm: int = 10000,
    seed: int = 42,
    fisher_z: bool = False,
    return_max_t: bool = True,
    return_fdr: bool = True,
    alternative: str = "two-sided",  # "two-sided", "greater", "less"
):
    """
    Nonparametric one-sample test per parcel using *sign flipping* across subjects.

    Parameters
    ----------
    corrs : DataFrame
        Rows = subjects, cols = parcels (RSA correlations per parcel).
    n_perm : int
        Number of permutations.
    seed : int
        RNG seed.
    fisher_z : bool
        If True, apply Fisher z (arctanh) to values before testing.
    return_max_t : bool
        If True, compute Westfall–Young max-|t| null for strong FWER control.
    return_fdr : bool
        If True, also return BH-FDR q-values based on permutation p's.
    alternative : {"two-sided","greater","less"}
        Tail of the test.

    Returns
    -------
    results : dict
        't_obs'           : Series of observed t statistics (per parcel)
        'p_perm'          : Series of permutation p-values (per parcel)
        'p_perm_maxT'     : Series of max-T FWER-corrected p-values (if return_max_t)
        'q_bh'            : Series of BH-FDR q-values from p_perm (if return_fdr)
    """
    rng = np.random.default_rng(seed)
    X = corrs.to_numpy(copy=True).astype(float)          # (n_subj, n_parc)
    if fisher_z:
        X = np.arctanh(X)  # stabilize variance for correlations in (-1, 1)

    n_subj, n_parc = X.shape
    # Observed t against 0
    t_obs, _ = ttest_1samp(X, popmean=0.0, axis=0, nan_policy='omit')

    # Precompute NaN-safe ingredients
    mask = ~np.isnan(X)                                  # valid entries
    n_eff = mask.sum(axis=0).astype(float)               # subjects per parcel
    X0 = np.where(mask, X, 0.0)                          # NaNs -> 0 for summation
    s = np.nanstd(X, axis=0, ddof=1)                     # sample SD per parcel
    denom = s / np.sqrt(np.maximum(n_eff, 1.0))          # t denominator
    safe = denom > 0

    # Helper to form tail-specific exceedances
    def exceed_null(t_null, t_obs_vec):
        if alternative == "two-sided":
            return (np.abs(t_null) >= np.abs(t_obs_vec)).astype(int)
        elif alternative == "greater":
            return (t_null >= t_obs_vec).astype(int)
        elif alternative == "less":
            return (t_null <= t_obs_vec).astype(int)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Permutation loop: sign-flips across subjects
    p_counts = np.zeros(n_parc, dtype=int)
    max_t_null = np.empty(n_perm) if return_max_t else None

    for b in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n_subj, 1))   # same sign for all parcels for a subject
        # mean after sign flip; NaN-safe via X0 and n_eff
        m_perm = (signs * X0).sum(axis=0) / np.maximum(n_eff, 1.0)
        t_null = np.zeros_like(m_perm)
        # compute t where denominator is defined
        t_null[safe] = m_perm[safe] / denom[safe]
        # tail-specific exceedances
        p_counts += exceed_null(t_null, t_obs)

        if return_max_t:
            max_t_null[b] = np.nanmax(np.abs(t_null))

    # Add-one smoothing for unbiased finite-B estimate
    p_perm = (p_counts + 1) / (n_perm + 1)

    results = {
        "t_obs": pd.Series(t_obs, index=corrs.columns),
        "p_perm": pd.Series(p_perm, index=corrs.columns),
    }

    if return_max_t:
        abs_t_obs = np.abs(t_obs)
        counts_max = (max_t_null[:, None] >= abs_t_obs[None, :]).sum(axis=0)
        p_perm_maxT = (counts_max + 1) / (n_perm + 1)
        results["p_perm_maxT"] = pd.Series(p_perm_maxT, index=corrs.columns)

    if return_fdr:
        # Benjamini–Hochberg on permutation p's
        p = results["p_perm"].to_numpy()
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(p) + 1)
        q = p * len(p) / ranks
        # enforce monotonicity
        q_sorted = np.minimum.accumulate(q[order[::-1]])[::-1]
        q_bh = np.empty_like(q)
        q_bh[order] = np.clip(q_sorted, 0, 1)
        results["q_bh"] = pd.Series(q_bh, index=corrs.columns)

    return results




### PLOTTING UTILS
# M.H.

def parc_list_to_surf(raw_values, n_parcs, mesh='FSAverage6'):

    # Given a list of length n_parcs with values to be plotted/shown on the brain, convert to two arrays of
    # length N surface nodes in each hemisphere, with proper values on each parcel.

    # This calls an annot file of "parcel to node mapppings" for your chosen surface space stored in Kraemer Lab discovery /parcellations
    # These come from schaefer atlas github, and fsaverage, fsaverage5, fsaverage6 are availabile
    # I tried to make my own mapping for fsaverage7 with script in /parcellations/Schaefer500_vol_to_surf.ipynb
    # But the parcel boundaries aren't super clean looking so probably best to just use fsaverage6
    
    # n_parcs = resolution of schaefer parcels to use
    # raw_values = scores to be plotted, one per parcel (ignored parcels must = 0)
    # mesh = what surface to use
    # returns two surface mesh objects, right and left hemispheres

    # ensure that your scores are floats
    values = [float(i) for i in raw_values]

    l_filename = '/dartfs-hpc/rc/lab/K/KraemerD/parcellations/schaefer/'+mesh+'_nodelabels/lh.Schaefer2018_500Parcels_7Networks_order.annot'
    r_filename = '/dartfs-hpc/rc/lab/K/KraemerD/parcellations/schaefer/'+mesh+'_nodelabels/rh.Schaefer2018_500Parcels_7Networks_order.annot'
    lh_labs = nibabel.freesurfer.io.read_annot(l_filename)[0]
    rh_labs = nibabel.freesurfer.io.read_annot(r_filename)[0]

    for node in range(len(rh_labs)): # need to adjust because parcels 251-500 are labeled 1-250 in this file
        if rh_labs[node] != 0: # is this node is included in a parcel
            rh_labs[node]+=250
    
    # check n parcs and len values to make sure they match
    if len(np.unique(list(lh_labs)+list(rh_labs)))-1 != len(values):
        print(str(len(values))+" values were given for ",str(len(np.unique(list(lh_labs)+list(rh_labs)))-1),"parcs...")

    # for each parcel, in each hemisphere, check surface nodes that belong to this parcel and replace them with that index value in parc_vals
    rh_masked = [0]*len(rh_labs)
    lh_masked = [0]*len(lh_labs)

    for p in range(1,len(values)+1):
        r_locs = [index for index, value in enumerate(rh_labs) if value == p]
        l_locs = [index for index, value in enumerate(lh_labs) if value == p]

        for r in r_locs:
            rh_masked[r] = values[p-1]

        for l in l_locs:
            lh_masked[l] = values[p-1]

    rh_masked = np.array(rh_masked)
    lh_masked = np.array(lh_masked)

    return rh_masked, lh_masked

def four_panel_surfplot(rh,lh,outfile,mesh='fsaverage6',title=' ', bg_on_data=False, cmap_method='center', custom_vmax=None, custom_vmin=None,threshold=0.0001,colormap='coolwarm',method='max'):

    # make, save, crop, and plot the inflated pial surface, showing left & right lateral & medial views
    # with a colorbar in the bottom right corner.
    # rh = right surface mesh with desired values (basically, an np array of len=n_nodes, 163842 for fsaverage7)
    # lh = left surface mesh
    # outfile = file name location for the output to be saved
    # mesh = what surface to use
    # title = string if you want a title printed at the top of the image

    # cmap_method = how should we color the results?
    #       'center' == center the colormap (so zero=the middle color, and pos/neg values are above & below, good for diverging maps like coolwarm)
    #       'range' == use the full range of the data & colorscale (lowest color = lowest value, highest color = highest value, to best utilize the full cmap)
    #       'custom' == use whatever values you manually set as custom_vmax and custom_vmin
    #       might add more methods as needed later

    # threshold, colormap, and method are params directly from nilearn's plot_surf

    # load surface
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)


    data_min = np.min([i for i in list(rh)+list(lh) if i != 0]) # lowest value that isn't zero
    data_max = np.max([i for i in list(rh)+list(lh) if i != 0]) # highest value that isn't zero

    # calculate vmin and vmax based on cmap_method:
    if cmap_method == 'center':
        vmax = sorted([abs(data_min),abs(data_max)],reverse=True)[0]
        vmin = -sorted([abs(data_max),abs(data_min)],reverse=True)[0]
    elif cmap_method == 'range':
        vmax = data_max
        vmin = data_min
    elif cmap_method == 'custom':
        vmax = custom_vmax
        vmin = custom_vmin

    else: print("cmap_method must equal 'center', 'range', or 'custom'")

    if vmax == vmin: # if this is a single parcel/binary map of 1s and 0s, etc.
        print("You're trying to use a map with only 1 value - use custom vmax and vmin instead")


    if threshold > abs(data_min):
        print("Your data values are smaller than the default threshold (0.0001), specify a lower threshold or some parcels may not be visible...")

    # Begin plotting - note one extra subplot column for the standalone colorbar
    fig, axs = plt.subplots(ncols=3,nrows=2,gridspec_kw={'width_ratios': [4,4,1]}, subplot_kw={'projection': '3d'},
                            figsize=(20,17),layout='constrained')

    plot_surf(fsaverage.infl_left, lh, hemi='left', threshold=threshold, view='lateral', cmap=colormap,
              bg_on_data=bg_on_data,vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_left,darkness=0.9,
              #antialiased=True, 
              avg_method=method, axes=axs[0,0])

    plot_surf(fsaverage.infl_right, rh, hemi='right',threshold=threshold, view='lateral', cmap=colormap,darkness=0.9,
                bg_on_data=bg_on_data,vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_right,
              #antialiased=True, 
              avg_method=method, axes=axs[0,1])

    plot_surf(fsaverage.infl_right, rh, hemi='right',threshold=threshold,view='medial',cmap=colormap,darkness=0.9,
              bg_on_data=bg_on_data,vmin=vmin,vmax=vmax,colorbar=False, bg_map=fsaverage.sulc_right,
              #antialiased=True, 
              avg_method=method, axes=axs[1,1])

    plot_surf(fsaverage.infl_left, lh, hemi='left',threshold=threshold, view='medial', cmap=colormap,darkness=0.9,
              bg_on_data=bg_on_data,vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_left,
              #antialiased=True, 
              avg_method=method, axes=axs[1,0])

    # hide unneeded subplot above the colorbar
    axs[0,2].axis('off')
    axs[1,2].axis('off')

    # # plot the colorbar (adding the colorbar to one of the figures makes them uneven sizes so here it's separate lol)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax,clip=True), cmap=colormap),
                 ax=axs[1,2],orientation='vertical',fraction=.2,pad=-1.2,aspect=10)
    cbar.ax.tick_params(labelsize=20)

    #when it's done, save it
    plt.savefig(outfile+'.png',facecolor='white')
    plt.close(fig)


    # This creates an image with a lot of white space! I tried very hard to deal with this by scaling the actual surfplots
    # But nilearn does some funny auto-scaling I couldn't override. So...just save the image, reload, and crop a column of white space out

    # we'll put the cropped images in a new folder called "cropped". first, create that if it doesn't exist
    outdir = outfile[:outfile.rfind('/')]
    outname = outfile[outfile.rfind('/')+1:]

    if not os.path.exists(outdir+'/cropped/'):
        os.makedirs(outdir+'/cropped/')

    new_outfile = outdir+'/cropped/'+outname

    with Image.open(outfile+'.png') as img:
        width, height = img.size
        print(width, height)
        
        # first, remove vertical column of white space between the hemispheres
        left_col = img.crop((0, 0, (width // 2) - (120),height-100)) # left column minus 120 pix
        right_col = img.crop(((width // 2)-20, 0, width, height))
        
        new_width = left_col.width + right_col.width
        new_height = height-100
        new_img = Image.new('RGB', (new_width, new_height))
        
        new_img.paste(left_col, (0, 0))
        new_img.paste(right_col, (left_col.width, 0))
        
        # then, remove the horizontal row of white space between the lateral & medial views
        top_row = new_img.crop((0, 0, new_width-60, (new_height // 2) - (130)))
        bottom_row = new_img.crop((0, (new_height // 2) + (120), new_width-60, new_height))
        
        # Create a new image to hold the top and bottom parts
        newer_height = top_row.height + bottom_row.height
        newer_img = Image.new('RGB', (new_width-100, newer_height))

        # Paste the top and bottom parts into the new image
        newer_img.paste(top_row, (0, 0))
        newer_img.paste(bottom_row, (0, top_row.height))
        
        # call draw method to add title text to the image
        draw = ImageDraw.Draw(newer_img)
        font = ImageFont.truetype("/dartfs-hpc/rc/lab/K/KraemerD/sharedconda/fonts/arial.ttf", 50)
        draw.text((50,20),str(title),fill= "black",font=font)

        draw.text((300,90),"LH",fill="black",font=font)
        draw.text((1300,90),"RH",fill="black",font=font)

        # Save the new image
        newer_img.save(new_outfile+'_cropped.png')


    return "Saved figure to "+new_outfile+"_cropped.png"



def dors_vent_surfplot(rh,lh,outfile,mesh='fsaverage6',title=' ', bg_on_data=False, cmap_method='center', custom_vmax=None, custom_vmin=None,threshold=0.0001,colormap='coolwarm',method='max'):

    # make, save, crop, and plot the inflated pial surface, showing left & right Dorsal and Ventral views
    # with a colorbar in the bottom right corner.
    # rh = right surface mesh with desired values (basically, an np array of len=n_nodes, 163842 for fsaverage7)
    # lh = left surface mesh
    # outfile = file name location for the output to be saved
    # mesh = what surface to use
    # title = string if you want a title printed at the top of the image

    # cmap_method = how should we color the results?
    #       'center' == center the colormap (so zero=the middle color, and pos/neg values are above & below, good for diverging maps like coolwarm)
    #       'range' == use the full range of the data & colorscale (lowest color = lowest value, highest color = highest value, to best utilize the full cmap)
    #       'custom' == use whatever values you manually set as custom_vmax and custom_vmin
    #       might add more methods as needed later

    # threshold, colormap, and method are params directly from nilearn's plot_surf

    # load surface
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)


    data_min = np.min([i for i in list(rh)+list(lh) if i != 0]) # lowest value that isn't zero
    data_max = np.max([i for i in list(rh)+list(lh) if i != 0]) # highest value that isn't zero

    # calculate vmin and vmax based on cmap_method:
    if cmap_method == 'center':
        vmax = sorted([abs(data_min),abs(data_max)],reverse=True)[0]
        vmin = -sorted([abs(data_max),abs(data_min)],reverse=True)[0]
    elif cmap_method == 'range':
        vmax = data_max
        vmin = data_min
    elif cmap_method == 'custom':
        vmax = custom_vmax
        vmin = custom_vmin

    else: print("cmap_method must equal 'center', 'range', or 'custom'")

    if vmax == vmin: # if this is a single parcel/binary map of 1s and 0s, etc.
        print("You're trying to use a map with only 1 value - use custom vmax and vmin instead")


    if threshold > abs(data_min):
        print("Your data values are smaller than the default threshold (0.0001), specify a lower threshold or some parcels may not be visible...")

    # Begin plotting - note one extra subplot column for the standalone colorbar
    fig, axs = plt.subplots(ncols=3,nrows=2,gridspec_kw={'width_ratios': [4,4,1]}, subplot_kw={'projection': '3d'},
                            figsize=(20,17),layout='constrained')

    plot_surf(fsaverage.infl_left, lh, hemi='left', threshold=threshold, view='dorsal', cmap=colormap,
              bg_on_data=bg_on_data,vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_left,
              #antialiased=True, 
              avg_method=method, axes=axs[0,0])

    plot_surf(fsaverage.infl_right, rh, hemi='right',threshold=threshold, view='dorsal', cmap=colormap,
                bg_on_data=bg_on_data,vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_right,
              #antialiased=True, 
              avg_method=method, axes=axs[0,1])

    plot_surf(fsaverage.infl_right, rh, hemi='right',threshold=threshold,view='ventral',cmap=colormap,
              bg_on_data=bg_on_data,vmin=vmin,vmax=vmax,colorbar=False, bg_map=fsaverage.sulc_right,
              #antialiased=True, 
              avg_method=method, axes=axs[1,1])

    plot_surf(fsaverage.infl_left, lh, hemi='left',threshold=threshold, view='ventral', cmap=colormap,
              bg_on_data=bg_on_data,vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_left,
              #antialiased=True, 
              avg_method=method, axes=axs[1,0])

    # hide unneeded subplot above the colorbar
    axs[0,2].axis('off')
    axs[1,2].axis('off')

    # # plot the colorbar (adding the colorbar to one of the figures makes them uneven sizes so here it's separate lol)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax,clip=True), cmap=colormap),
                 ax=axs[1,2],orientation='vertical',fraction=.2,pad=-1.2,aspect=10)
    cbar.ax.tick_params(labelsize=20)

    #when it's done, save it
    plt.savefig(outfile+'.png',facecolor='white')
    plt.close(fig)


    # This creates an image with a lot of white space! I tried very hard to deal with this by scaling the actual surfplots
    # But nilearn does some funny auto-scaling I couldn't override. So...just save the image, reload, and crop a column of white space out

    # we'll put the cropped images in a new folder called "cropped". first, create that if it doesn't exist
    outdir = outfile[:outfile.rfind('/')]
    outname = outfile[outfile.rfind('/')+1:]

    if not os.path.exists(outdir+'/cropped/'):
        os.makedirs(outdir+'/cropped/')

    new_outfile = outdir+'/cropped/'+outname

    with Image.open(outfile+'.png') as img:
        width, height = img.size
        print(width, height)
        
        # first, remove vertical column of white space between the hemispheres
        left_col = img.crop((0, 0, (width // 2) - (120),height-100)) # left column minus 120 pix
        right_col = img.crop(((width // 2)-20, 0, width, height))
        
        new_width = left_col.width + right_col.width
        new_height = height-100
        new_img = Image.new('RGB', (new_width, new_height))
        
        new_img.paste(left_col, (0, 0))
        new_img.paste(right_col, (left_col.width, 0))
        
        # then, remove the horizontal row of white space between the lateral & medial views
        top_row = new_img.crop((0, 0, new_width-60, (new_height // 2) - (130)))
        bottom_row = new_img.crop((0, (new_height // 2) + (120), new_width-60, new_height))
        
        # Create a new image to hold the top and bottom parts
        newer_height = top_row.height + bottom_row.height
        newer_img = Image.new('RGB', (new_width-100, newer_height))

        # Paste the top and bottom parts into the new image
        newer_img.paste(top_row, (0, 0))
        newer_img.paste(bottom_row, (0, top_row.height))
        
        # call draw method to add title text to the image
        draw = ImageDraw.Draw(newer_img)
        font = ImageFont.truetype("/dartfs-hpc/rc/lab/K/KraemerD/sharedconda/fonts/arial.ttf", 50)
        draw.text((50,20),str(title),fill= "black",font=font)

        draw.text((300,90),"LH",fill="black",font=font)
        draw.text((1300,90),"RH",fill="black",font=font)

        # Save the new image
        newer_img.save(new_outfile+'_cropped.png')


    return "Saved figure to "+new_outfile+"_cropped.png"

