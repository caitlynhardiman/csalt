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



par_names = ['Inc', 'Mstar', 'h', 'r_c', 'psi', 'PA', 'Dust α', 'Vturb', 'Mgas', 'gd ratio']
prec = [0.1, 0.01, 0.1, 0.1, 0.001, 0.1, 0.0001, 0.001, 0.0001, 0.1]
lbls = ['data', 'model', 'residual']


tclean_kw = {'imsize': 1500, 'start': '4.5km/s', 'width': '0.5km/s',
             'nchan': 7, 'restfreq': '345.796GHz', 'cell': '0.01arcsec',
             'scales': [0, 5, 15, 35, 100], 'niter': 1000,
             'robust': 1.0, 'threshold': '3mJy', 'uvtaper': '0.04arcsec',
             'gridder': 'mosaic', 'perchanweightdensity': False}  
        

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
    os.environ["OMP_NUM_THREADS"] = "1"
    mdict = cm.modeldict(ddict, params, kwargs=fixed_kw)
    os.environ["OMP_NUM_THREADS"] = "8"
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



def reimage_model(params, data, directory, fixed_kwargs, likelihood=None, iteration=None, modeltype='MCFOST'):

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

def best_model(h5file, data, directory, fixed_kwargs, visibility=True, params='all'):

    filename = h5file
    reader = emcee.backends.HDFBackend(filename, read_only=True)
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
    if visibility:
        make_plot(best_params, data, directory, fixed_kwargs, likelihood=best_likelihood, iteration=i)
    else:
        make_plot_image_plane(best_params, imagecube='/home/chardima/runs/DM_Tau_12CO_beam0.15_28ms_3sigma.clean.image.fits', iteration=i, fixed_kwargs=fixed_kwargs, param_names=params)
    
    return

def make_plot_image_plane(params, imagecube, iteration, fixed_kwargs, param_names):

    # Make model using best params - can use parametric disc?
    pfile = 'parametric_disk_MCFOST'
    if not os.path.exists(pfile+'.py'):
        print('The prescription '+pfile+'.py does not exist.  Exiting.')
        sys.exit()
    pd = importlib.import_module(pfile) 

    cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15.fits'
    if not os.path.exists(cropped_cube):
        data = casa.Cube(imagecube)
        data.cutout(filename=cropped_cube, FOV=15, vmin=4, vmax=8)
    data = casa.Cube(cropped_cube)

    check_para_matches_data('csalt.para', data)

    if param_names == 'all':
        inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio = params
        velax = np.array([4500, 5000, 5500, 6000, 6500, 7000, 7500])
        cfg_dict = fixed_kwargs['cfg_dict']
        vsyst = cfg_dict['vsyst']
        model, directory = pd.write_run_mcfost(velax, inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio, vsyst, ozstar=False)
    elif param_names == 'no vturb':
        inc, m, h, rc, psi, PA, dust_a, gas_mass, gasdust_ratio = params
        velax = np.array([4500, 5000, 5500, 6000, 6500, 7000, 7500])
        cfg_dict = fixed_kwargs['cfg_dict']
        vsyst = cfg_dict['vsyst']
        model, directory = pd.write_run_mcfost(velax, inc, m, h, rc, psi, PA, dust_a, gas_mass, gasdust_ratio, vsyst, ozstar=False)
    else:
        print('not recognised!')
        return 


    beam_area = data.bmin * data.bmaj * np.pi / (4.0 * np.log(2.0))
    pix_area = data.pixelscale**2
    datalines = data.image
    #datalines = data.image * pix_area/beam_area
    datalines[np.isnan(datalines)] = 0
    velocities = model.velocity
    data_chans = []
    model_chans = []
    flux_data = []
    flux_model = []
    for vel in velocities:
        iv = np.abs(data.velocity - vel).argmin()
        data_chans.append(datalines[iv])
        flux_data.append(np.sum(datalines[iv]))
        iv_m = np.abs(model.velocity - vel).argmin()
        print(model.velocity[iv_m])
        model_chans.append(model.lines[iv_m])
        flux_model.append(np.nansum(model.lines[iv_m]))
    
    residuals = mcfost.Line(directory+'/data_CO')
    residuals.lines = np.array(data_chans) #-0*np.array(model_chans)
    #beam_for_model = np.sqrt((residuals.pixelscale/data.pixelscale)**2 * data.bmin * data.bmaj)

    print(model.pixelscale)
    print(residuals.pixelscale)
    print(data.pixelscale)

    # Plotting arguments
    fmax = 0.05
    cmap = 'Blues'
    fmin = 0
    colorbar = False
    vlabel_color = 'black'
    lim = 6.99
    limits = [lim, -lim, -lim, lim]
    no_ylabel = False

    # Make plot
    fig, axs = plt.subplots(3, 7, figsize=(17, 6), sharex='all', sharey='all')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(7):
        if i != 0:
            no_ylabel = True
        if i == 6:
            colorbar = True
        if i != 3:
            no_xlabel = True
        else:
            no_xlabel = False
        data.plot(ax=axs[0, i], v=velocities[i], fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, no_vlabel=False, vlabel_color='black', limits=None, no_xlabel=True, no_ylabel=True)
        axs[0, i].get_xaxis().set_visible(False)
        model.plot_map(ax=axs[1, i], v=velocities[i],  bmaj=data.bmaj, bmin=data.bmin, bpa=data.bpa, fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, per_beam=True, limits=None, no_xlabel=True, no_ylabel=no_ylabel, no_vlabel=False, no_xticks=True)
        convolved = model.last_image
        residuals.lines[i] = np.array(data_chans[i]) - np.array(convolved)
        residuals.plot_map(ax=axs[2, i], v=velocities[i],  fmin=-fmax, fmax=fmax, cmap='RdBu', colorbar=colorbar, limits=None, no_ylabel=True, no_vlabel=False, no_xlabel=no_xlabel)


    fig.suptitle('Visibility Image Plane Best '+str(iteration))
    plt.savefig("imageplane_best_"+str(iteration)+".png", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)

    figure = plt.figure()
    plt.plot(velocities, flux_data, label='data')
    plt.plot(velocities, flux_model, label='model')
    plt.xlabel('Velocities (km/s)')
    plt.ylabel('Flux (Jy)')
    plt.title('Line profile - Iteration '+str(iteration))
    plt.savefig('lineprofile'+str(iteration)+'.png', dpi=300)
    plt.clf()

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
    reader = emcee.backends.HDFBackend(filename, read_only=True)
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


def best_100_models(h5file, fixed_kwargs, params='all'): 

    filename = h5file
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    all_samples = reader.get_chain(discard=0, flat=True)
    logpost_samples = reader.get_log_prob(discard=0, flat=True)

    sorted_unique, unique_indices = np.unique(logpost_samples, return_index=True)
    top_100 = sorted_unique[-100:]
    top_100_index = unique_indices[-100:]

    parameters = []
    likelihoods = []

    for i in range(100):
        index = top_100_index[i]
        parameters.append(all_samples[index])
        likelihoods.append(logpost_samples[index])

    for i in range(100):
        make_plot_image_plane(params=parameters[i], imagecube='/home/chardima/runs/DM_Tau_12CO_beam0.15_28ms_3sigma.clean.image.fits', iteration=i, fixed_kwargs=fixed_kwargs, param_names=params)


    return


def check_para_matches_data(parafile, data):

    para = mcfost.Params(parafile)
    model_FOV = para.map.size/para.map.distance
    if data.FOV != model_FOV:
        size = data.FOV*para.map.distance
        print('Size should be '+str(size)+' but was '+str(para.map.size))
        para.map.size = size

    if data.nx != para.map.nx:
        print('Npix should be '+str(data.nx)+' but was '+str(para.map.nx))
        para.map.nx = data.nx
        para.map.ny = data.ny

    para.writeto(parafile)



def find_best_params(h5file):

    reader = emcee.backends.HDFBackend(h5file, read_only=True)
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

    return best_params, best_likelihood, i

def line_prof_image_plane(params, param_names, likelihood, iteration, directory, prec, vsyst=6.04, imagecube='/home/chardima/runs/DM_Tau_12CO_beam0.15_28ms_3sigma.clean.image.fits'):

    cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15.fits'
    if not os.path.exists(cropped_cube):
        data = casa.Cube(imagecube)
        data.cutout(filename=cropped_cube, FOV=15, vmin=4, vmax=8)
    data = casa.Cube(cropped_cube)

    check_para_matches_data('csalt.para', data)

    pfile = 'parametric_disk_MCFOST'
    if not os.path.exists(pfile+'.py'):
        print('The prescription '+pfile+'.py does not exist.  Exiting.')
        sys.exit()
    pd = importlib.import_module(pfile) 

    if param_names == 'all':
        param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 'flaring_exp', 
                       'PA', 'dust_param', 'vturb', 'gas_mass', 'gasdust_ratio']
    params = convert_to_dict(param_names, params)

    print(params)

    velocities = np.array([4500, 5000, 5500, 6000, 6500, 7000, 7500])
    model, model_directory = pd.write_run_mcfost(velax=velocities, vsyst=vsyst, ozstar=False, **params)

    # Line profile
    data_lp = data.get_line_profile()
    model_lp = np.sum(model.lines[:, :, :], axis=(1, 2))

    figure = plt.figure()
    plt.plot(data.velocity, data_lp, label='data')
    plt.plot(model.velocity, model_lp, label='model')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux (Jy)')
    plt.title('Best Model Line Profile')
    plt.savefig(directory+'/lineprofile'+str(iteration)+'.png', dpi=300)
    plt.clf()


    # Image Plane Plot

    residuals = mcfost.Line(model_directory+'/data_CO')

    # Plotting arguments
    fmax = 0.05
    cmap = 'Blues'
    fmin = 0
    colorbar = False
    vlabel_color = 'black'
    no_ylabel = False
    velocities = velocities/1000

    # Make plot
    fig, axs = plt.subplots(3, 7, figsize=(16, 6), sharex='all', sharey='all')
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for i in range(7):
        if i != 0:
            no_ylabel = True
            no_yticks = True
        if i == 6:
            colorbar = True
        if i != 3:
            no_xlabel = True
        else:
            no_xlabel = False
        data.plot(ax=axs[0, i], v=velocities[i], fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, no_vlabel=False, vlabel_color='black', limits=None, no_xlabel=True, no_ylabel=True)
        axs[0, i].get_xaxis().set_visible(False)
        model.unit='Jy/beam'
        residuals.unit='Jy/beam'
        model.plot_map(ax=axs[1, i], v=velocities[i],  bmaj=data.bmaj, bmin=data.bmin, bpa=data.bpa, fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, per_beam=True, limits=None, no_xlabel=True, no_ylabel=no_ylabel, no_vlabel=False, no_xticks=True)
        convolved = model.last_image
        data_index = np.argmin(np.abs(data.velocity - velocities[i]))
        residuals.lines[i] = np.array(data.image[data_index]) - np.array(convolved)
        residuals.plot_map(ax=axs[2, i], v=velocities[i],  fmin=-fmax, fmax=fmax, cmap='RdBu', colorbar=colorbar, limits=None, no_ylabel=True, no_vlabel=False, no_xlabel=no_xlabel)
        if i != 0:
            for j in range(3):
                axs[j, i].get_yaxis().set_visible(False)

    my_text = '- Parameters -'

    for i in range(len(param_names)):
        dec_places = np.abs(int(np.log10(prec[i])))
        if dec_places >= 4:
            my_text += '\n' + param_names[i] + ' = ' + ('{:.'+str(1)+'e}').format(params[param_names[i]])
        else:
            my_text += '\n' + param_names[i] + ' = ' + ('{:.'+str(dec_places)+'f}').format(params[param_names[i]])

    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    axs[0][0].text(-1.98, 0.2, my_text, transform=axs[0][0].transAxes, fontsize=9, horizontalalignment='left', verticalalignment='center', bbox=props)
    #plt.tight_layout()

    fig.suptitle('Best Model in Image Plane - Likelihood = '+str(likelihood))
    plt.savefig(directory+"/imageplane_best_"+str(iteration)+".png", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)



def convert_to_dict(names, values):

    param_dict = {}
    for i in range(len(names)):
        param_dict[names[i]] = values[i]
    
    return param_dict