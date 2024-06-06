import os
import sys
import numpy as np
from .model import *
from .helpers import *
from .fit_mcfost import *
from functools import partial
from casatasks import exportfits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.io import fits
import scipy.constants as sc
import cmasher as cmr

import casa_cube as casa


lbls = ['data', 'model', 'residual']
lbls_5 = ['exoALMA cube', 'imaged visibility model', 'imaged visibility residual', 'image plane model', 'image plane residual']

# Define some tclean keywords to be used in all imaging
tclean_kw = {'imsize': 1500, 'start': '4.5km/s', 'width': '0.5km/s',
             'nchan': 7, 'restfreq': '345.796GHz', 'cell': '0.01arcsec',
             'scales': [0, 5, 15, 35, 100], 'niter': 1000,
             'robust': 1.0, 'threshold': '3mJy', 'uvtaper': '0.04arcsec',
             'gridder': 'mosaic', 'perchanweightdensity': False}

# Define some Keplerian mask keywords
kepmask_kw = {'inc': 36.6, 'PA': 155, 'mstar': 0.4, 'dist': 144.5, 'vlsr': 6100, 
              'r_max': 1.1, 'nbeams': 1.5, 'zr': 0.2}


data = '/home/chardima/runs/FITTING_EXPERIMENTS/concatenated.ms'

fixed_kwargs = {'FOV': 6.375, 'Npix': 256, 'dist': 144.5, 'restfreq':345.796e9, 'vsyst': 6.04}


def image_data(data):

    imagecube(data, 'DATA', kepmask_kwargs=kepmask_kw, tclean_kwargs=tclean_kw)
    exportfits('DATA.image', fitsimage='DATA.fits', velocity=True, overwrite=True)

    return



def image_model(params, data, directory, fixed_kw, modeltype, dicts=False):

    val = ''
    for key, value in params.items():
        val += '_'+str(value)

    if not dicts:
        # Read in the data MS
        ddict = read_MS(data)
        # Instantiate a csalt model
        cm = model(modeltype)
        # Calculate a model dictionary; write it to model and residual MS files
        print("Model Dictionary")
        fixed_kw['directory'] = directory
        mdict = cm.modeldict(ddict, params, kwargs=fixed_kw)
        if directory is None:
            directory = '.'
        write_MS(mdict, outfile=directory+'/model/MODEL'+str(val)+'.ms')
        write_MS(mdict, outfile=directory+'/residual/RESID'+str(val)+'.ms', resid=True)
        print('Made dictionary!')

    origin = os.getcwd()
    os.chdir(directory)

    # Image model, and residual cubes - move to individual directories
    imagecube('model/MODEL'+str(val)+'.ms', 'model/MODEL'+str(val), 
                mk_kepmask=False, tclean_kwargs=tclean_kw)
    imagecube('residual/RESID'+str(val)+'.ms', 'residual/RESID'+str(val),
            mk_kepmask=False, tclean_kwargs=tclean_kw)
    os.chdir(origin)
    
    print('model and residual imaged!')

    # Export the cubes to FITS format
    cubes = ['DATA', directory+'/model/MODEL'+str(val), directory+'/residual/RESID'+str(val)]

    for i in range(1, 3):
        exportfits(cubes[i]+'.image', fitsimage=cubes[i]+'.fits', 
                    velocity=True, overwrite=True) 
        
    return cubes

def make_plot(data, param, val, posterior, all_values, all_posteriors, fixed_kw):

    # set up a temporary directory for the ms files to be in
    jobfs = os.getenv("JOBFS")
    print(jobfs)
    jobfs = None
    if jobfs == None:
        jobfs = ''
    directory = jobfs+param+'_'+str(val)


    if not os.path.isfile('DATA.fits'):
        image_data(data)

    # directory to put plot into
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/model/')
        os.mkdir(directory+'/residual/')

    params = {}
    params[param] = val

    cubes = image_model(params, data, directory, fixed_kw, modeltype='MCFOST')

    plot_name = construct_plot(param, cubes, val, posterior, all_values, all_posteriors)

    return

def make_plot_five(vis_data, im_data, param, val, vis_posterior, im_posterior, 
                   all_values, all_vis_posteriors, all_im_posteriors, fixed_kw):

    # set up a temporary directory for the ms files to be in
    jobfs = os.getenv("JOBFS")
    print(jobfs)
    #jobfs = None
    if jobfs == None:
        jobfs = ''
    directory = jobfs+param+'_'+str(val)


    if not os.path.isfile('DATA.fits'):
        image_data(data)

    # directory to put plot into
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/model/')
        os.mkdir(directory+'/residual/')

    params = {}
    params[param] = val

    cubes = image_model(params, vis_data, directory, fixed_kw, modeltype='MCFOST')

    data_cube = im_data
    im_model = directory+'/data_CO/'
    im_resid = directory+'/data_CO/'
    vis_model = cubes[1]
    vis_resid = cubes[2]

    cubes = [data_cube, im_model, im_resid, vis_model, vis_resid]

    plot_name = construct_plot_lineprofile_five(param, cubes, val, vis_posterior, im_posterior, 
                                                all_values, all_vis_posteriors, all_im_posteriors)

    return



def construct_plot(param, cubes, val, posterior, values, posteriors):

    # Plot a subset of the cube as a direct comparison
    nchans = 7

    fig, axs = plt.subplots(nrows=3, ncols=nchans+5, figsize=(15.4, 3.5))
    fl, fr, fb, ft, hs, ws = 0.22, 0.9, 0.12, 0.905, 0.12, 0.03
    xlims = [2.5, -2.5]
    ylims = [-2.5, 2.5]

    gs = axs[0, 0].get_gridspec()  # Access the subplot in the top left corner
    # remove the underlying axes
    for ax in axs[0:, 0:5].flatten()[1:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, 0:4])
    axbig.plot(values, posteriors)
    axbig.scatter(val, posterior)
    axbig.set_xlabel(param)
    axbig.set_ylabel('Posteriors')
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])

    # Cubes

    for ia in range(len(cubes)):

        # Load the cube and header data
        hdu = fits.open(cubes[ia]+'.fits')
        Ico, h = np.squeeze(hdu[0].data), hdu[0].header
        hdu.close()

        # coordinate grids, beam parameters
        dx = 3600 * h['CDELT1'] * (np.arange(h['NAXIS1']) - (h['CRPIX1'] - 1))
        dy = 3600 * h['CDELT2'] * (np.arange(h['NAXIS2']) - (h['CRPIX2'] - 1))
        ext = (dx.max(), dx.min(), dy.min(), dy.max())
        bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
        bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi / 180)**2

        # spectral information
        vv = h['CRVAL3'] + h['CDELT3'] * (np.arange(h['NAXIS3']) - (h['CRPIX3']-1))
        print(vv)
        ff = 345.796e9 * (1 - vv / sc.c)

        for i in range(nchans):

            # in-cube index
            j = i + 0

            # convert intensities to brightness temperatures
            Tb = (1e-26 * np.squeeze(Ico[j,:,:]) / bm) * sc.c**2 / \
                    (2 * sc.k * ff[j]**2)

            # allocate the panel
            ax = axs[ia, i+5]
            pax = ax.get_position()

            # plot the channel map
            vmin = 0
            vmax = 80
            cmap = 'cmr.cosmic'
            if ia == 2:
                vmin = -vmax
                cmap = 'cmr.redshift'
            im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=ext,
                            aspect='auto', vmin=vmin, vmax=vmax)

            # set map boundaries
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            # overlay beam dimensions
            beam = Ellipse((xlims[0] + 0.1*np.diff(xlims),
                            -xlims[0] - 0.1*np.diff(xlims)), 
                            3600 * bmaj, 3600 * bmin, angle=90-bpa)
            beam.set_facecolor('w')
            ax.add_artist(beam)

            # labeling
            if i == 0:
                ax.text(0.02, 0.90, lbls[ia], transform=ax.transAxes, ha='left',
                        va='center', style='italic', fontsize=8, color='w')
            if ia == 2:
                if np.abs(vv[j]) < 0.001: vv[j] = 0.0
                if np.logical_or(np.sign(vv[j]) == 1, np.sign(vv[j]) == 0):
                    pref = '+'
                else:
                    pref = ''
                vstr = pref+'%.2f' % (1e-3 * vv[j])
                ax.text(0.97, 0.08, vstr, transform=ax.transAxes, ha='right',
                        va='center', fontsize=7, color='w')
            if np.logical_and(ia == 2, i == 0):
                ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
                ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            else:
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    # colorbar
    cbax = fig.add_axes([fr+0.01, fb+0.01, 0.02, ft-fb-0.02])
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                    ticklocation='right')
    cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=13)

    plt.tight_layout()

    fig.suptitle(param+' = '+str(val)+', likelihood = '+str(posterior))
    fig.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, hspace=hs, wspace=ws)
    plot_name = param+'_'+str(val)+'_plot.png'
    fig.savefig(plot_name)
    fig.clf()

    return plot_name



def construct_plot_lineprofile_five(param, cubes, val, vis_posterior, im_posterior, 
                                                values, vis_posteriors, im_posteriors):

    # Plot a subset of the cube as a direct comparison
    nchans = 7

    fig, axs = plt.subplots(nrows=8, ncols=nchans, figsize=(10, 12))
    fl, fr, fb, ft, hs, ws = 0.08, 0.9, 0.06, 0.905, 0.12, 0.03
    xlims = [2.5, -2.5]
    ylims = [-2.5, 2.5]

    gs = axs[0, 0].get_gridspec()
    for ax in axs[5:, 0:].flatten()[0:]:
        ax.remove()

    ax_likelihood = fig.add_subplot(gs[6:, 0:3])
    ax_lineprofile = fig.add_subplot(gs[6:, 4:])
    
    ax_likelihood.plot(values, vis_posteriors, color='red')
    ax_likelihood.scatter([val], [vis_posterior], color='red')
    ax_likelihood.tick_params(axis='y', labelcolor='red')
    ax_likelihood.set_title('Log posterior as a function of ' + param)
    ax_likelihood.set_xlabel(param)
    ax_likelihood.set_ylabel('Vis log posterior', color='red')
    ax_likelihood2 = ax_likelihood.twinx()
    ax_likelihood2.tick_params(axis='y', labelcolor='blue')
    ax_likelihood2.plot(values, im_posteriors, color='blue')
    ax_likelihood2.scatter([val], [im_posterior], color='blue')
    ax_likelihood2.set_ylabel('Im log posterior', color='blue')

    #line profiles
    data = casa.Cube(cubes[0])
    beam_area = data.bmin * data.bmaj * np.pi / (4.0 * np.log(2.0))
    pix_area = data.pixelscale**2
    velocities = data.velocity
    datalines = data.image * pix_area/beam_area
    datalines[np.isnan(datalines)] = 0

    flux_data = []
    flux_model = []

    for chan in datalines:
        flux_data.append(np.sum(chan))
    for chan in modellines:
        flux_model.append(np.sum(chan))

    ax_lineprofile.plot(velocities, flux_data, label='Image Data')
    ax_lineprofile.plot(velocities, flux_model, label='Image Model')
    ax_lineprofile.plot(vis_velocities, flux_vis_data, label='Vis Data')
    ax_lineprofile.plot(vis_velocities, flux_vis_model, label='Vis Model')
    ax_lineprofile.set_title('Line Profiles')
    ax_lineprofile.set_xlabel('Velocity (km/s)')
    ax_lineprofile.set_ylabel('Flux (Jy)')

    # Cubes

    for ia in range(len(cubes)):

        # Load the cube and header data
        hdu = fits.open(cubes[ia]+'.fits')
        Ico, h = np.squeeze(hdu[0].data), hdu[0].header
        hdu.close()

        # coordinate grids, beam parameters
        dx = 3600 * h['CDELT1'] * (np.arange(h['NAXIS1']) - (h['CRPIX1'] - 1))
        dy = 3600 * h['CDELT2'] * (np.arange(h['NAXIS2']) - (h['CRPIX2'] - 1))
        ext = (dx.max(), dx.min(), dy.min(), dy.max())
        bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
        bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi / 180)**2

        # spectral information
        vv = h['CRVAL3'] + h['CDELT3'] * (np.arange(h['NAXIS3']) - (h['CRPIX3']-1))
        print(vv)
        ff = 345.796e9 * (1 - vv / sc.c)

        for i in range(nchans):

            # in-cube index
            j = i + 0

            # convert intensities to brightness temperatures
            Tb = (1e-26 * np.squeeze(Ico[j,:,:]) / bm) * sc.c**2 / \
                    (2 * sc.k * ff[j]**2)

            # allocate the panel
            ax = axs[ia, i]
            pax = ax.get_position()

            # plot the channel map
            vmin = 0
            vmax = 80
            cmap = 'cmr.cosmic'
            if ia == 2:
                vmin = -vmax
                cmap = 'cmr.redshift'
            im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=ext,
                            aspect='auto', vmin=vmin, vmax=vmax)

            # set map boundaries
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            # overlay beam dimensions
            beam = Ellipse((xlims[0] + 0.1*np.diff(xlims),
                            -xlims[0] - 0.1*np.diff(xlims)), 
                            3600 * bmaj, 3600 * bmin, angle=90-bpa)
            beam.set_facecolor('w')
            ax.add_artist(beam)

            # labeling
            if i == 0:
                ax.text(0.02, 0.90, lbls_5[ia], transform=ax.transAxes, ha='left',
                        va='center', style='italic', fontsize=8, color='w')
            if ia == 2:
                if np.abs(vv[j]) < 0.001: vv[j] = 0.0
                if np.logical_or(np.sign(vv[j]) == 1, np.sign(vv[j]) == 0):
                    pref = '+'
                else:
                    pref = ''
                vstr = pref+'%.2f' % (1e-3 * vv[j])
                ax.text(0.97, 0.08, vstr, transform=ax.transAxes, ha='right',
                        va='center', fontsize=7, color='w')
            if np.logical_and(ia == 2, i == 0):
                ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
                ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            else:
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    # colorbar
    cbax = fig.add_axes([fr+0.01, 0.3825, 0.02, ((ft-fb)/8 * 5)-0.01])
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                    ticklocation='right')
    cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=13)

    plt.tight_layout()

    fig.suptitle(param+' = '+str(val)+', vis likelihood = '+str(vis_posterior))
    fig.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, hspace=hs, wspace=ws)
    plot_name = param+'_'+str(val)+'_plot.png'
    fig.savefig(plot_name)
    fig.clf()

    return plot_name


###########################################################################################


# We are passing in to this script the parameter we are making the movie of, the values and likelihoods and the one we are specifically making the movie of here


# param = sys.argv[1]
# index = sys.argv[2]

# results_file = np.load(param+'_results.npz')
# all_values = results_file['arr_0']
# all_posteriors = results_file['arr_1']

# val = all_values[index]
# posterior = all_posteriors[index]

# make_plot(data, param, val, posterior, all_values, all_posteriors)