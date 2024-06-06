import os
import sys
import numpy as np
from .model import *
from .helpers import *
from .fit_mcfost import *
import multiprocess
from functools import partial
from casatasks import exportfits
import emcee

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.io import fits
import scipy.constants as sc
import cmasher as cmr

#  def make_movie(self, backend_file, original_ms, vra_fit=[5.75e3, 6.25e3], nwalk=75, nsteps=100, ninits=10, nthreads=64, niter=100):

#     def likelihood(self, params):
#         return self.m.get_probability(self.infdata, params)

par_names = ['Inc', 'Mstar', 'h', 'r_c', 'psi', 'PA', 'Dust α', 'Vturb', 'Mgas', 'gd ratio']
prec = [0.1, 0.01, 0.1, 0.1, 0.001, 0.1, 0.0001, 0.001, 0.0001, 0.1]
lbls = ['data', 'model', 'residual']

# Define some tclean keywords to be used in all imaging
tclean_kw = {'imsize': 1500, 'start': '4.5km/s', 'width': '0.5km/s',
             'nchan': 7, 'restfreq': '345.796GHz', 'cell': '0.01arcsec',
             'scales': [0, 10, 30, 60], 'niter': 0,
             'robust': 1.0, 'threshold': '3mJy', 'uvtaper': '0.04arcsec'}

# tclean_kw = {'imsize': 1024, 'start': '4.5km/s', 'width': '0.5km/s',
#              'nchan': 7, 'restfreq': '345.796GHz', 'cell': '0.0135arcsec',
#              'scales': [0, 10, 30, 60], 'niter': 100,
#              'robust': 1.0, 'threshold': '3mJy', 'uvtaper': '0.04arcsec'}   

print(tclean_kw)          

# Define some Keplerian mask keywords
kepmask_kw = {'inc': 36.6, 'PA': 155, 'mstar': 0.4, 'dist': 144.5, 'vlsr': 6100, 
              'r_max': 1.1, 'nbeams': 1.5, 'zr': 0.2}


def image_data(data):

    imagecube(data, 'DMTau_DATA', kepmask_kwargs=kepmask_kw, tclean_kwargs=tclean_kw)
    exportfits('DMTau_DATA.image', fitsimage='DMTau_DATA.fits', velocity=True, overwrite=True)

    return


def image_model(params, data, fixed_kw, directory, modeltype):

    pool_id = multiprocess.current_process()
    pool_id = str(pool_id.pid)
    # Read in the data MS
    ddict = read_MS(data)
    # Instantiate a csalt model
    cm = model(modeltype)
    # Calculate a model dictionary; write it to model and residual MS files
    print("Model Dictionary")
    mdict = cm.modeldict(ddict, params, kwargs=fixed_kw)
    write_MS(mdict, outfile=directory+'/model/DMTau_MODEL_'+pool_id+'.ms')
    write_MS(mdict, outfile=directory+'/residual/DMTau_RESID_'+pool_id+'.ms', resid=True)
    print('Made dictionary!')

    # Image model, and residual cubes
    imagecube(directory+'/model/DMTau_MODEL_'+pool_id+'.ms', directory+'/model/DMTau_MODEL'+pool_id, 
                mk_kepmask=False, tclean_kwargs=tclean_kw)
    imagecube(directory+'/residual/DMTau_RESID_'+pool_id+'.ms', directory+'/residual/DMTau_RESID'+pool_id,
            mk_kepmask=False, tclean_kwargs=tclean_kw)
    
    print('model and residual imaged!')

    # Export the cubes to FITS format
    cubes = ['DMTau_DATA', directory+'/model/DMTau_MODEL'+pool_id, directory+'/residual/DMTau_RESID'+pool_id]

    for i in range(1, 3):
        exportfits(cubes[i]+'.image', fitsimage=cubes[i]+'.fits', 
                    velocity=True, overwrite=True) 
        
    return cubes


def get_file_info(backend_file, mode, niter, walker):

    filename = backend_file
    reader = emcee.backends.HDFBackend(filename)
    all_samples = reader.get_chain(discard=0, flat=False)
    logpost_samples = reader.get_log_prob(discard=0, flat=False)

    model_params = []
    model_likelihoods = []
    iterations = []

    if mode == 'best':
        for iteration in range(0, len(all_samples), niter):
            iterations.append(iteration)
            best_model_logprob = -np.inf
            for walker in logpost_samples[iteration]:
                if walker > best_model_logprob:
                    best_model_logprob = walker
            best_index = np.argwhere(logpost_samples[iteration]==best_model_logprob)
            model_likelihoods.append(logpost_samples[iteration][best_index])
            model_params.append(all_samples[iteration][best_index])
    elif mode == 'walker':
        if walker == None:
            walker = 0
        for iteration in range(0, len(all_samples), niter):
            iterations.append(iteration)
            logprob = logpost_samples[iteration][walker]
            params = all_samples[iteration][walker]
            model_likelihoods.append(logprob)
            model_params.append(params)
    else:
        print('Unrecognized mode - choose either by walker or best per iteration')
        return 0
    
    return model_params, model_likelihoods, iterations

def check_model_likelihoods(params, likelihoods, nthreads, original_ms, vra_fit, nwalk, nsteps, ninits):


    # Checking likelihood - skip for now
    m = setup_fit(msfile=original_ms, vra_fit=vra_fit, nwalk=nwalk, nsteps=nsteps, ninits=ninits)
    infdata = m.initialise()

    os.environ["OMP_NUM_THREADS"] = "1"

    from multiprocessing.pool import Pool

    p = Pool(nthreads)

    with p:
        calculated_likelihoods = p.map(likelihood, params)
    p.close()

    for walker in range(len(likelihoods)):
        print('From file: ', likelihoods[walker])
        print('Calculated: ', calculated_likelihoods[walker])
        if round(likelihoods[walker], 2) != round(calculated_likelihoods[walker], 2):
            print("Likelihoods don't match!")
            return 0
    
    return


def make_movie(backend_file, data, mode, directory, fixed_kwargs, niter=1, walker=None, nthreads=16):

    if not os.path.isfile('DMTau_DATA.fits'):
        image_data(data)

    model_params, model_likelihoods, iterations = get_file_info(backend_file, mode, niter, walker)

    #check_model_likelihoods(model_params, model_likelihoods) 

    make_plot_partial = partial(make_plot_formovie, data=data, directory=directory, fixed_kwargs=fixed_kwargs)


    theta = list(zip(model_params, model_likelihoods, iterations))

    os.environ["OMP_NUM_THREADS"] = "1"


    with multiprocessing.get_context("spawn").Pool(processes=nthreads) as pool:
        plot_names = pool.map(make_plot_partial, theta)

    # then make movie with plot_names

    return



def make_plot(params, data, directory, fixed_kwargs, likelihood=None, iteration=None, modeltype='MCFOST'):

    # if likelihood is None:
    #     likelihood = get_likelihood(params, data)

    if not os.path.isfile('DMTau_DATA.fits'):
        image_data(data)

    # directory to put plot into
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/model/')
        os.mkdir(directory+'/residual/')

    cubes = image_model(params, data, fixed_kwargs, directory, modeltype)

    plot_name = construct_plot(params, cubes, likelihood, directory, iteration)

    return plot_name


def make_plot_formovie(theta, data, directory, fixed_kwargs):

    params, likelihood, iteration = theta

    if not os.path.isfile('DMTau_DATA.fits'):
        image_data(data)

    # directory to put plot into
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/model/')
        os.mkdir(directory+'/residual/')

    cubes = image_model(params, data, fixed_kwargs, directory)

    plot_name = construct_plot(params, cubes, likelihood, directory, iteration)

    return plot_name

def construct_plot(pars, cubes, likelihood, directory, iteration):

    # Plot a subset of the cube as a direct comparison
    nchans = 7

    fig, axs = plt.subplots(nrows=3, ncols=nchans, figsize=(9, 3.5))
    fl, fr, fb, ft, hs, ws = 0.22, 0.9, 0.12, 0.905, 0.12, 0.03
    xlims = [2.5, -2.5]
    ylims = [-2.5, 2.5]

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

    my_text = '- Parameters -'

    for i in range(len(pars)):
        dec_places = np.abs(int(np.log10(prec[i])))
        if dec_places >= 4:
            my_text += '\n' + par_names[i] + ' = ' + ('{:.'+str(1)+'e}').format(pars[i])
        else:
            my_text += '\n' + par_names[i] + ' = ' + ('{:.'+str(dec_places)+'f}').format(pars[i])

    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    axs[0][0].text(-1.98, 0.2, my_text, transform=axs[0][0].transAxes, fontsize=9, horizontalalignment='left', verticalalignment='center', bbox=props)
    plt.tight_layout()

    fig.suptitle('Likelihood = ' + str(likelihood))
    fig.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, hspace=hs, wspace=ws)
    plot_name = directory+'/model_comparison_'+str(iteration)+'.pdf'
    fig.savefig(plot_name)
    print('plot made')
    fig.clf()

    return plot_name

def best_model(h5file, data, directory, fixed_kwargs):

    filename = h5file
    reader = emcee.backends.HDFBackend(filename)
    all_samples = reader.get_chain(discard=0, flat=False)
    logpost_samples = reader.get_log_prob(discard=0, flat=False)
    best_model_logprob = -np.inf

    for iteration in range(len(all_samples)):
        for walker in logpost_samples[iteration]:
            if walker > best_model_logprob:
                best_model_logprob = walker
    best_index = np.argwhere(logpost_samples==best_model_logprob)
    if len(best_index) > 1:
        best_index = best_index[-1]
    i, j = best_index

    best_params = all_samples[i][j]
    best_likelihood = logpost_samples[i][j]
    print(best_params)

    # check that I can retrieve it

    # make plot
    make_plot(best_params, data, directory, fixed_kwargs, likelihood=best_likelihood, iteration=i)
    return


def worst_model(h5file, data, directory, fixed_kwargs):

    filename = h5file
    reader = emcee.backends.HDFBackend(filename)
    all_samples = reader.get_chain(discard=0, flat=False)
    logpost_samples = reader.get_log_prob(discard=0, flat=False)
    worst_model_logprob = 0

    for iteration in range(len(all_samples)):
        for walker in logpost_samples[iteration]:
            if walker < worst_model_logprob:
                worst_model_logprob = walker
    worst_index = np.argwhere(logpost_samples==worst_model_logprob)
    i, j = worst_index[-1]

    worst_params = all_samples[i][j]
    worst_likelihood = logpost_samples[i][j]
    print(worst_params)

    # check that I can retrieve it

    # make plot
    make_plot(worst_params, data, directory, fixed_kwargs, likelihood=worst_likelihood, iteration=i)

    return

def median_model(h5file, data, directory, fixed_kwargs):

    filename = h5file
    reader = emcee.backends.HDFBackend(filename)
    all_samples = reader.get_chain(discard=0, flat=False)
    latest = all_samples[-1]
    latest= np.transpose(latest)
    median_params = []
    for param in latest:
        median_params.append(np.median(param))
    
    print(median_params)

    # check that I can retrieve it

    # make plot
    make_plot(median_params, data, directory, fixed_kwargs)
    return
