# useful functions for plotting results on the fsaverage6 surface

import os
import numpy as np
import copy
import pickle
import nilearn.datasets as datasets
from nilearn import surface
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf
from PIL import Image, ImageDraw, ImageFont


def parc_list_to_surf(raw_values, n_parcs, mesh='FSAverage7'):

    # Given a list of length n_parcs with values to be plotted/shown on the brain, convert to two arrays of
    # length N surface nodes in each hemisphere, with proper values on each parcel.

    # This calls a pickled array of "parcel to node mapppings" stored in Kraemer Lab discovery /parcellations
    # Must create this array for your chosen parcellation + surface mesh with Schaefer500_vol_to_surf.ipynb first

    # n_parcs = resolution of schaefer parcels to use
    # raw_values = scores to be plotted, one per parcel (ignored parcels must = 0)
    # mesh = what surface to use
    # returns two surface mesh objects, right and left hemispheres

    # ensure that your scores are floats
    values = [float(i) for i in raw_values]

    nodelabels = pickle.load(open('/dartfs-hpc/rc/lab/K/KraemerD/parcellations/schaefer/Schaefer2018_'+str(n_parcs)+'Parcels_7Networks_order_'+mesh+'_nodelabels.pkl','rb'))

    # check n parcs and len values to make sure they match
    if len(np.unique(list(nodelabels[0])+list(nodelabels[1])))-1 != len(values):
        print(str(len(values))+" values were given for ",str(len(np.unique(listnodelabels[0])+list(nodelabels[1]))-1),"parcs...")

    # for each parcel, in each hemisphere, check surface nodes that belong to this parcel and replace them with that index value in parc_vals
    rh_masked = [0]*len(nodelabels[0])
    lh_masked = [0]*len(nodelabels[1])

    for p in range(1,len(values)+1):
        r_locs = [index for index, value in enumerate(nodelabels[0]) if value == p]
        l_locs = [index for index, value in enumerate(nodelabels[1]) if value == p]

        for r in r_locs:
            rh_masked[r] = values[p-1]

        for l in l_locs:
            lh_masked[l] = values[p-1]

    rh_masked = np.array(rh_masked)
    lh_masked = np.array(lh_masked)

    return rh_masked, lh_masked

def four_panel_surfplot(rh,lh,outfile,mesh='fsaverage7',title=' ', cmap_method='center', custom_vmax=None, custom_vmin=None,threshold=0.0001,colormap='coolwarm',method='max'):

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
              vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_left,
              antialiased=True, avg_method=method, axes=axs[0,0])

    plot_surf(fsaverage.infl_right, rh, hemi='right',threshold=threshold, view='lateral', cmap=colormap,
              vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_right,
              antialiased=True, avg_method=method, axes=axs[0,1])

    plot_surf(fsaverage.infl_right, rh, hemi='right',threshold=threshold,view='medial',cmap=colormap,
              vmin=vmin,vmax=vmax,colorbar=False, bg_map=fsaverage.sulc_right,
              antialiased=True, avg_method=method, axes=axs[1,1])

    plot_surf(fsaverage.infl_left, lh, hemi='left',threshold=threshold, view='medial', cmap=colormap,
              vmin=vmin, vmax=vmax, colorbar=False, bg_map=fsaverage.sulc_left,
              antialiased=True, avg_method=method, axes=axs[1,0])

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





