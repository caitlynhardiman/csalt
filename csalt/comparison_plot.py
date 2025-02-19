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


tclean_kw = {'imsize': 1500, 'start': '4.5km/s', 'width': '0.028km/s',
             'nchan': 108, 'restfreq': '345.796GHz', 'cell': '0.01arcsec',
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

    print('Imaging model...')

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
    write_MS(mdict, outfile=directory+'/DMTau_MODEL.ms')
    write_MS(mdict, outfile=directory+'/DMTau_RESID.ms', resid=True)
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
        print('Imaging data...')
        image_data(data)

    # directory to put plot into
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.exists(directory+'/model/'):
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

#     my_text = '- Parameters -'

# #    for i in range(len(pars)):
# #         dec_places = np.abs(int(np.log10(prec[i])))
# #         if dec_places >= 4:
# #             my_text += '\n' + par_names[i] + ' = ' + ('{:.'+str(1)+'e}').format(pars[i])
# #         else:
# #             my_text += '\n' + par_names[i] + ' = ' + ('{:.'+str(dec_places)+'f}').format(pars[i])

# #     props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
# #      axs[0][0].text(-1.98, 0.2, my_text, transform=axs[0][0].transAxes, fontsize=9, horizontalalignment='left', verticalalignment='center', bbox=props)
#     plt.tight_layout()

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
        reimage_model(best_params, data, directory, fixed_kwargs, likelihood=best_likelihood, iteration=i)
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

    velax = np.array([4500, 5000, 5500, 6000, 6500, 7000, 7500])
    cfg_dict = fixed_kwargs['cfg_dict']
    vsyst = cfg_dict['vsyst']

    if param_names == 'all':
        inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio = params
        model, directory = pd.write_run_mcfost(velax, inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio, vsyst, ozstar=False)
    elif param_names == 'no vturb':
        inc, m, h, rc, psi, PA, dust_a, gas_mass, gasdust_ratio = params
        model, directory = pd.write_run_mcfost(velax, inc, m, h, rc, psi, PA, dust_a, gas_mass, gasdust_ratio, vsyst, ozstar=False)
    elif param_names == 'extra':
        inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio, teff, mu_RA, mu_DEC = params
        model, directory = pd.write_run_mcfost(velax, inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio, teff, mu_RA, mu_DEC, vsyst, ozstar=False)
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
    plt.savefig("imageplane_best_"+str(iteration)+".pdf", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)

    figure = plt.figure()
    plt.plot(velocities, flux_data, label='data')
    plt.plot(velocities, flux_model, label='model')
    plt.xlabel('Velocities (km/s)')
    plt.ylabel('Flux (Jy)')
    #plt.title('Line profile - Iteration '+str(iteration))
    plt.savefig('lineprofile'+str(iteration)+'.pdf', dpi=300)
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


def best_100_models(h5file, fixed_kwargs=None, params='all'): 

    filename = h5file
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    all_samples = reader.get_chain(discard=0, flat=True)
    logpost_samples = reader.get_log_prob(discard=0, flat=True)

    sorted_unique, unique_indices = np.unique(logpost_samples, return_index=True)
    top_10000 = sorted_unique[-10000:]
    top_10000_index = unique_indices[-10000:]

    parameters = []
    likelihoods = []

    for i in range(10000):
        index = top_10000_index[i]
        parameters.append(all_samples[index])
        likelihoods.append(logpost_samples[index])

    best_file = open("best_10000.txt", "w")
    for i in range(10000):
        for j in range(len(parameters[i])):
            number = parameters[i][j]
            number_reformatted = '{:.3g}'.format(number)
            best_file.write(number_reformatted+' ')
        best_file.write(str(likelihoods[i])+'\n')
    best_file.close()

    #for i in range(100):
    #    make_plot_image_plane(params=parameters[i], imagecube='/home/chardima/runs/DM_Tau_12CO_beam0.15_28ms_3sigma.clean.image.fits', iteration=i, fixed_kwargs=fixed_kwargs, param_names=params)


    return parameters


# def models_above_1sigma(h5file): 

#     filename = h5file
#     reader = emcee.backends.HDFBackend(filename, read_only=True)
#     all_samples = reader.get_chain(discard=0, flat=True)
#     logpost_samples = reader.get_log_prob(discard=0, flat=True)

#     sorted_unique, unique_indices = np.unique(logpost_samples, return_index=True)
#     threshold_likelihood = 
#     indices_above_1sigma = np.argwhere(sorted_unique > threshold_likelihood)
#     log(Lmax) - 1.15

#     top_1000 = sorted_unique[-1000:]
#     top_1000_index = unique_indices[-1000:]

#     parameters = []
#     likelihoods = []

#     for i in range(1000):
#         index = top_1000_index[i]
#         parameters.append(all_samples[index])
#         likelihoods.append(logpost_samples[index])

#     best_file = open("best_1000.txt", "w")
#     for i in range(1000):
#         for j in range(len(parameters[i])):
#             number = parameters[i][j]
#             print(number)
#             number_reformatted = '{:.3g}'.format(number)
#             best_file.write(number_reformatted+' ')
#         best_file.write(str(likelihoods[i])+'\n')
#     best_file.close()


#     return



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
    best_index = best_index[-1]
    i, j = best_index

    best_params = all_samples[i][j]
    best_likelihood = logpost_samples[i][j]

    return best_params, best_likelihood, i

def line_prof_image_plane(params, param_names, likelihood, iteration, directory, prec, eb3=False, vsyst=6.04, imagecube='/home/chardima/runs/imaging_eb3/all_ebs.fits_robust1.0_27ms_4sigma.clean.image.fits', disk='DM_Tau'):

    if disk == 'DM_Tau':
        if eb3:
            cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15_eb3.fits'
            fmax = 0.99
        else:
            cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15.fits'
            fmax = 0.048
    elif disk == 'V4046':
        cropped_cube = '/home/chardima/runs/V4046_FITTING/V4046_Sgr_cutout_FOV_15.fits'
        fmax = 0.048
    #cropped_cube = '/home/chardima/runs/DM_Tau_reimaged_cutout_FOV_15.fits'
    if not os.path.exists(cropped_cube):
        data = casa.Cube(imagecube)
        data.cutout(filename=cropped_cube, FOV=15, vmin=4, vmax=8)
    data = casa.Cube(cropped_cube)
    beamarea = np.pi * data.bmaj * data.bmin / (4.0 * np.log(2.0))
    pixarea = data.pixelscale**2
    rms = 0.0008412888 # Jy/beam
    rms = 0.0050633242 # Jy/beam

    check_para_matches_data('csalt.para', data)

    pfile = 'parametric_disk_MCFOST'
    if not os.path.exists(pfile+'.py'):
        print('The prescription '+pfile+'.py does not exist.  Exiting.')
        sys.exit()
    pd = importlib.import_module(pfile) 

    if param_names == 'all':
        param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 'flaring_exp', 
                       'PA', 'dust_param', 'vturb', 'gas_mass', 'gasdust_ratio']
    elif param_names == 'extra':
        param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 'flaring_exp', 
                       'PA', 'dust_param', 'vturb', 'gas_mass', 'gasdust_ratio', 'teff', 'mu_RA', 'mu_DEC']
    elif param_names == 'co_df':
        param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 'flaring_exp', 
                       'PA', 'dust_param', 'vturb', 'gas_mass', 'gasdust_ratio', 'co_abundance', 'depletion_factor']
    elif param_names == 'gammas':
        param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 'flaring_exp', 
                       'PA', 'dust_param', 'vturb', 'gas_mass', 'gasdust_ratio', 'co_abundance', 'depletion_factor', 'gamma', 'gamma_exp']        
    
    params = convert_to_dict(param_names, params)

    print(params)

    velax = data.velocity*1000
    model, model_directory = pd.write_run_mcfost(velax=velax, vsyst=vsyst, ozstar=False, **params)

    # Line profile
    data_lp = data.get_line_profile()
    model_lp = np.sum(model.lines[:, :, :], axis=(1, 2))

    # residuals_for_line_profile = mcfost.Line(model_directory+'/data_CO')
    # print('residuals for line profile')
    # convolved = []
    # for i in range(len(residuals_for_line_profile.lines)):
    #     residuals_for_line_profile.plot_map(iv=i, bmaj=data.bmaj, bmin=data.bmin, bpa=data.bpa, per_beam=True)
    #     convolved.append(residuals_for_line_profile.last_image)
    #     plt.close()
    #     data_index = np.argmin(np.abs(data.velocity - model.velocity[i]))
    #     residuals_for_line_profile.lines[i] = np.array(data.image[data_index]) - np.array(convolved[i])
    # beam_area = np.pi * data.bmaj * data.bmin / (4.0 * np.log(2.0))
    # pix_area = residuals_for_line_profile.pixelscale ** 2
    # residual_lp = np.sum(residuals_for_line_profile.lines[:, :, :], axis=(1, 2)) / (beam_area/pix_area)


    figure = plt.figure()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.plot(data.velocity, data_lp, label='all baselines', color='purple')
    if not eb3 and disk=='DM_Tau':
        eb3_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15_eb3.fits'
        eb3_data = casa.Cube(eb3_cube)
        eb3_data_lp = eb3_data.get_line_profile()
        plt.plot(eb3_data.velocity, eb3_data_lp, label='short baseline', color='deeppink')
    plt.plot(model.velocity, model_lp, label='best model', color='orange')
    # plt.plot(residuals_for_line_profile.velocity, residual_lp, label='residual')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux (Jy)')
    #plt.title('Best Model Line Profile')
    plt.legend(prop={'size': 10})
    plt.savefig(directory+'/lineprofile_best_'+str(iteration)+'.pdf', dpi=300)
    plt.clf()



    figure = plt.figure()
    origin = os.getcwd()
    os.chdir(model_directory)
    model_th = mcfost.SED('data_th')
    model_th.plot_T(Tmin=0, Tmax=80, cmap='RdYlBu_r', linear_colorbar=True)
    os.chdir(origin)
    plt.savefig(directory+'/temperature'+str(iteration)+'.png', dpi=300)
    plt.clf()


    # Image Plane Plot

    residuals = mcfost.Line(model_directory+'/data_CO')

    # Plotting arguments
    
    cmap = 'Blues'
    fmin = 0
    colorbar = False
    vlabel_color = 'black'
    no_ylabel = False
    velocities = np.array([vsyst-1.5, vsyst-1, vsyst-0.5, vsyst, vsyst+0.5, vsyst+1, vsyst+1.5])
    #velocities = np.linspace(5.5, 6.5, 7)

    # Make plot
    fig, axs = plt.subplots(3, 7, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    contour_level = 3*rms
    for i in range(7):
        data.plot(ax=axs[0, i], v=velocities[i], fmin=-fmax, fmax=fmax, cmap='RdBu', colorbar=False, no_vlabel=False, vlabel_color='black', limits=None, no_xlabel=True, no_ylabel=True)
        data.plot(ax=axs[0, i], v=velocities[i], cmap='RdBu', colorbar=False, no_vlabel=False, vlabel_color='black', limits=None, no_xlabel=True, no_ylabel=True, plot_type='contour', levels=[contour_level])
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].set_yticklabels([])
        if i == 0:
            axs[0, 0].text(
                0.05, 0.85,
                "Data",
                horizontalalignment='left',
                fontsize=10, 
                color="black",
                transform=axs[0, i].transAxes,
            ) 
        # For model and residual plots
        for j in range(1, 3):  # j=1 is model, j=2 is residual
            if j == 2 and i == 0:  # Show labels only for first column of residual row
                no_ylabel = False
                no_xlabel = False
            else:
                no_ylabel = True
                no_xlabel = True

            if j == 1:  # Model plot
                model.unit='Jy/beam'
                mod = model.plot_map(ax=axs[j, i], v=velocities[i], bmaj=data.bmaj, bmin=data.bmin, bpa=data.bpa, 
                                   fmin=-fmax, fmax=fmax, cmap='RdBu', colorbar=False, per_beam=True, 
                                   limits=None, no_xlabel=no_xlabel, no_ylabel=no_ylabel, no_vlabel=True, no_xticks=True)
                convolved = model.last_image
                if i == 0:
                    axs[1, 0].text(
                        0.05, 0.85,
                        "Model",
                        horizontalalignment='left',
                        fontsize=10, 
                        color="black",
                        transform=axs[1, i].transAxes,
                    )
            else:  # Residual plot
                residuals.unit='Jy/beam'
                data_index = np.argmin(np.abs(data.velocity - velocities[i]))
                residuals.lines[data_index] = np.array(data.image[data_index]) - np.array(convolved)
                resid = residuals.plot_map(ax=axs[j, i], v=velocities[i], fmin=-fmax/3, fmax=fmax/3, 
                                         cmap='RdBu', colorbar=False, limits=None, 
                                         no_ylabel=no_ylabel, no_vlabel=True, no_xlabel=no_xlabel)
                if i == 0:
                    axs[2, 0].text(
                        0.05, 0.85,
                        "Data - Model",
                        horizontalalignment='left',
                        fontsize=10, 
                        color="black",
                        transform=axs[2, i].transAxes,
                         bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')  # Added bbox for semi-transparent box
                    )

            if i != 0 or j != 2:  # Remove tick labels except for first column of residual row
                axs[j, i].set_yticklabels([])
                axs[j, i].set_xticklabels([])

    unit = 'Jy/beam'
    p = np.array([ax.get_position().get_points().flatten() for ax in axs.ravel()])
    xmin = np.amin(p[:,0]) ; xmax = np.amax(p[:,2]) ; dx = xmax - xmin
    ymin = np.amin(p[:,1]) ; ymax = np.amax(p[:,3]) ; dy = ymax - ymin
    gap = 0.01 * dy
    ymax2 = ymin + (1/3)*dy - gap/2 ; dy2 = ymax2 - ymin
    ymin1 = ymax2 + gap ; dy1 = ymax - ymin1
    shift = 0.01
    width = 0.01
    cax1 = fig.add_axes([xmax + shift*dx, ymin1, width * dx, dy1])
    cax2 = fig.add_axes([xmax + shift*dx, ymin, width * dx, dy2])
    cax1.xaxis.set_ticks_position('top')
    cax2.xaxis.set_ticks_position('top')
    cb1 = fig.colorbar(mod, cax=cax1, orientation="vertical")       
    cb1.set_label(unit, size=12)
    cb2 = fig.colorbar(resid, cax=cax2, orientation="vertical")       
    cb2.set_label(unit, size=12)

    # my_text = '- Parameters -'

    # for i in range(len(param_names)):
    #     if prec is not None:
    #         dec_places = np.abs(int(np.log10(prec[i])))
    #         if dec_places >= 4:
    #             my_text += '\n' + param_names[i] + ' = ' + ('{:.'+str(1)+'e}').format(params[param_names[i]])
    #         else:
    #             my_text += '\n' + param_names[i] + ' = ' + ('{:.'+str(dec_places)+'f}').format(params[param_names[i]])
    #     else:
    #         if params[param_names[i]] is not None:
    #             my_text += '\n' + param_names[i] + ' = ' + str(round(params[param_names[i]], 3))

    # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    # axs[0][0].text(-1.9, 0.1, my_text, transform=axs[0][0].transAxes, fontsize=9, horizontalalignment='left', verticalalignment='center', bbox=props)
    # #plt.tight_layout()
    # if likelihood is not None:
    #     fig.suptitle('Best Model in Image Plane - Likelihood = '+str(likelihood))
    plt.savefig(directory+"/imageplane_best_"+str(iteration)+".pdf", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)


def convert_to_dict(names, values):

    param_dict = {}
    for i in range(len(names)):
        param_dict[names[i]] = values[i]
    
    return param_dict


def multi_line_prof_image_plane(params, param_names, likelihood, iteration, directory, prec, vsyst=6.04, imagecube='/home/chardima/runs/imaging_eb3/all_ebs.fits_robust1.0_27ms_4sigma.clean.image.fits'):

    print('changed!!')
    #cropped_cube_full = '/home/chardima/runs/DM_Tau_cutout_FOV_15.fits'
    cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15_eb3.fits'
    #cropped_cube = '/home/chardima/runs/DM_Tau_reimaged_cutout_FOV_15.fits'
    if not os.path.exists(cropped_cube):
        data = casa.Cube(imagecube)
        data.cutout(filename=cropped_cube, FOV=15, vmin=4, vmax=8)
    data = casa.Cube(cropped_cube)
    beamarea = np.pi * data.bmaj * data.bmin / (4.0 * np.log(2.0))
    pixarea = data.pixelscale**2
    rms = 0.0008412888 # Jy/beam

    #data_full = casa.Cube(cropped_cube_full)

    check_para_matches_data('csalt.para', data)

    pfile = 'parametric_disk_MCFOST'
    if not os.path.exists(pfile+'.py'):
        print('The prescription '+pfile+'.py does not exist.  Exiting.')
        sys.exit()
    pd = importlib.import_module(pfile) 

    parameter_sets = []
    models = []
    residuals_group = []

    for model_param_set in params:
        model_param_set = convert_to_dict(param_names, model_param_set)
        parameter_sets.append(model_param_set)

    velax = data.velocity*1000
    for model_param_set in parameter_sets:
        model, model_directory = pd.write_run_mcfost(velax=velax, vsyst=vsyst, ozstar=False, **model_param_set)
        models.append(model)
        residuals = mcfost.Line(model_directory+'/data_CO')
        residuals_group.append(residuals)

    masses = []
    inclinations = []
    vturb = []

    for model_param_set in params:
        masses.append(round(model_param_set[1], 2))
        inclinations.append(int(round(model_param_set[0], 2)))
        vturb.append(int(round(model_param_set[7]*1000, 2)))

    # Line profile
    data_lp = data.get_line_profile()
    #data_lp_full = data_full.get_line_profile()
    model_lps = []
    for model in models:
        model_lp = np.sum(model.lines[:, :, :], axis=(1, 2))
        model_lps.append(model_lp)


    figure = plt.figure()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    colors = ['red', 'deeppink', 'orange']

    print(len(models), len(model_lps))

    plt.plot(data.velocity, data_lp, label='data', color='purple')
    #plt.plot(data_full.velocity, data_lp_full, label='full_cube', color='blue')
    for i in range(len(models)):
        print(i)
        label = str(masses[i])+r' M$_\odot$'+', '+str(inclinations[i])+r'$^\circ$'+', '+str(vturb[i])+r' m/s'
        plt.plot(models[i].velocity, model_lps[i], label=label, color=colors[i])
    # plt.plot(residuals_for_line_profile.velocity, residual_lp, label='residual')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux (Jy)')
    #plt.title('Best Model Line Profile')
    plt.legend(prop={'size': 7})
    plt.savefig(directory+'/lineprofile_multi_'+str(iteration)+'.pdf', dpi=300)
    plt.clf()

    print('done!')


    # Image Plane Plot


    # Plotting arguments
    fmax = 0.99
    #fmax = 0.048
    cmap = 'Blues'
    fmin = 0
    colorbar = False
    vlabel_color = 'black'
    no_ylabel = True
    velocities = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    #velocities = np.linspace(5.5, 6.5, 7)

    n_models = len(models)

    # Make plot
    fig, axs = plt.subplots(1+(1*n_models), 7, figsize=(14, 2*(1+1*n_models)))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(7):
        data.plot(ax=axs[0, i], v=velocities[i], fmin=0, fmax=fmax, cmap='Blues', colorbar=False, no_vlabel=False, vlabel_color='black', limits=None, no_xlabel=True, no_ylabel=True)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].set_yticklabels([])

        if i == 0:
            axs[0, i].text(
                0.05, 0.85,
                "Data",
                horizontalalignment='left',
                fontsize=10, 
                color="black",
                transform=axs[0, i].transAxes,
            ) 

        ax_num = 1
        for model_number in range(n_models):
            model_to_plot = models[model_number]
            model_to_plot.unit='Jy/beam'
            if model_number == 2 and i == 0:
                no_ylabel = False
                no_xlabel = False
            else:
                no_ylabel = True
                no_xlabel = True
            test = model_to_plot.plot_map(ax=axs[ax_num, i], v=velocities[i],  bmaj=data.bmaj, bmin=data.bmin, bpa=data.bpa, fmin=0, fmax=fmax, cmap='Blues', colorbar=False, per_beam=True, limits=None, no_xlabel=no_xlabel, no_ylabel=no_ylabel, no_vlabel=True) 
            if i == 0:
                label = str(masses[model_number])+r' M$_\odot$'+'\n'+str(inclinations[model_number])+r'$^\circ$'+'\n'+str(vturb[model_number])+r' m/s'
                axs[ax_num, i].text(
                    0.05, 0.65,
                    label,
                    fontsize=10, 
                    horizontalalignment='left',
                    color="black",
                    transform=axs[ax_num, i].transAxes,
                )  

            if i != 0 or model_number != 2:
                axs[ax_num, i].set_yticklabels([])
                axs[ax_num, i].set_xticklabels([])              
            
            ax_num+=1    

    unit = 'Jy/beam'
    p = np.array([ax.get_position().get_points().flatten() for ax in axs.ravel()])
    xmin = np.amin(p[:,0]) ; xmax = np.amax(p[:,2]) ; dx = xmax - xmin
    ymin = np.amin(p[:,1]) ; ymax = np.amax(p[:,3]) ; dy = ymax - ymin
    shift = 0.01
    width = 0.01
    cax = fig.add_axes([xmax + shift*dx, ymin, width * dx, dy])
    cax.xaxis.set_ticks_position('top')
    cb = fig.colorbar(test, cax=cax, orientation="vertical")       
    cb.set_label(unit, size=12)
    

    plt.savefig(directory+"/multiplot_"+str(iteration)+".pdf", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)




def testing(params, param_names, likelihood, iteration, directory, prec, vsyst=6.04, imagecube='/home/chardima/runs/imaging_eb3/all_ebs.fits_robust1.0_27ms_4sigma.clean.image.fits'):

    print('did change')
    #cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15.fits'
    cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15_eb3.fits'
    #cropped_cube = '/home/chardima/runs/DM_Tau_reimaged_cutout_FOV_15.fits'
    data = casa.Cube(cropped_cube)

    # Image Plane Plot

    # Plotting arguments
    fmax = 0.99
    #fmax = 0.048
    cmap = 'Blues'
    fmin = 0
    colorbar = False
    vlabel_color = 'black'
    no_ylabel = True
    velocities = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    #velocities = np.linspace(5.5, 6.5, 7)

    masses = [0.6, 0.8, 0.9]
    inclinations = [37, 31, 28]
    vturb = [73, 75, 90]

    # Make plot
    fig, axs = plt.subplots(1+(1*3), 7, figsize=(14, 2*(1+1*3)))#, sharex='all', sharey='all')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(7):
        data.plot(ax=axs[0, i], v=velocities[i], fmin=0, fmax=fmax, cmap='Blues', colorbar=False, no_vlabel=False, vlabel_color='black', limits=None, no_xlabel=True, no_ylabel=True)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].set_yticklabels([])

        if i == 0:
            axs[0, i].text(
                0.05, 0.85,
                "Data",
                horizontalalignment='left',
                fontsize=10, 
                color="black",
                transform=axs[0, i].transAxes,
            ) 
        #residuals.unit='Jy/beam'
        ax_num = 1
        for model_number in range(3):
            if model_number == 2 and i == 0:
                no_ylabel = False
                no_xlabel = False
            else:
                no_ylabel = True
                no_xlabel = True
            test = data.plot(ax=axs[ax_num, i], v=velocities[i], fmin=0, fmax=fmax, cmap='Blues', colorbar=False, no_vlabel=True, vlabel_color='black', limits=None, no_xlabel=no_xlabel, no_ylabel=no_ylabel)
            if i == 0:
                label = str(masses[model_number])+r' M$_\odot$'+'\n'+str(inclinations[model_number])+r'$^\circ$'+'\n'+str(vturb[model_number])+r' km s$^{-1}$'
                axs[ax_num, i].text(
                    0.05, 0.65,
                    label,
                    fontsize=10, 
                    horizontalalignment='left',
                    color="black",
                    transform=axs[ax_num, i].transAxes,
                )      

            if i != 0 or model_number != 2:
                axs[ax_num, i].set_yticklabels([])
                axs[ax_num, i].set_xticklabels([])

            ax_num+=1

    unit = 'Jy/beam'
    p = np.array([ax.get_position().get_points().flatten() for ax in axs.ravel()])
    xmin = np.amin(p[:,0]) ; xmax = np.amax(p[:,2]) ; dx = xmax - xmin
    ymin = np.amin(p[:,1]) ; ymax = np.amax(p[:,3]) ; dy = ymax - ymin
    shift = 0.01
    width = 0.01
    cax = fig.add_axes([xmax + shift*dx, ymin, width * dx, dy])
    cax.xaxis.set_ticks_position('top')
    cb = fig.colorbar(test, cax=cax, orientation="vertical")
    cb.set_label(unit, size=12)

        

    plt.savefig(directory+"/test"+str(iteration)+".png", bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)



def check_fluxes(h5file, param_names, nthreads, level): 

    reader = emcee.backends.HDFBackend(h5file, read_only=True)
    all_samples = reader.get_chain(discard=0, flat=True)
    logpost_samples = reader.get_log_prob(discard=0, flat=True)

    sorted_unique, unique_indices = np.unique(all_samples, axis=0, return_index=True)

    small_vturbs = []
    medium_small_vturbs = []
    medium_vturbs = []
    large_vturbs = []

    for i in range(1, len(sorted_unique)+1):
        parameters = all_samples[unique_indices[-i]]
        vturb = parameters[7]
        if vturb < 0.05:
            small_vturbs.append(unique_indices[-i])
        elif vturb > 0.05 and vturb < 0.095:
            medium_small_vturbs.append(unique_indices[-i])
        elif vturb > 0.1 and vturb < 0.15:
            medium_vturbs.append(unique_indices[-i])
        elif vturb > 0.2:
            large_vturbs.append(unique_indices[-i])

    # small_f = open("small_vturbs.txt", "x")
    # for index in small_vturbs:
    #     small_f.write(str(index)+'\n')
    # small_f.close()

    # medium_f = open("medium_vturbs.txt", "x")
    # for index in medium_vturbs:
    #     medium_f.write(str(index)+'\n')
    # medium_f.close()

    # large_f = open("large_vturbs.txt", "x")
    # for index in large_vturbs:
    #     large_f.write(str(index)+'\n')
    # large_f.close()

    if level == 'small':
        indices_to_check = small_vturbs
    elif level == 'medium':
        indices_to_check = medium_vturbs
    elif level == 'large':
        indices_to_check = large_vturbs
    else:
        print('Specify level to check!')
        return

    parameter_set = np.array(all_samples[indices_to_check])
    model_param_sets = []
    for param_set in parameter_set:
        model_param_set = convert_to_dict(param_names, param_set)
        model_param_sets.append(model_param_set)
    
    from multiprocessing import Pool

    with Pool(processes=nthreads) as pool:
        fluxes = pool.map(calc_flux, model_param_sets)

    print(len(fluxes))

    filename = level+'_vturb_fluxes.npz'
    np.savez(filename, parameter_set, fluxes)

    return fluxes


def calc_flux(model_param_set):

    cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15_eb3.fits'
    data = casa.Cube(cropped_cube)
    beamarea = np.pi * data.bmaj * data.bmin / (4.0 * np.log(2.0))
    pixarea = data.pixelscale**2
    rms = 0.0008412888 # Jy/beam

    check_para_matches_data('csalt.para', data)
    data_lp = data.get_line_profile()

    data_channels = [63, 76, 88]
    data_fluxes = [data_lp[i] for i in data_channels] # [20.486362, 13.579187, 19.092772]
    data_vels = [data.velocity[i] for i in data_channels]

    velax = [round(i*1000) for i in data_vels]

    pfile = 'parametric_disk_MCFOST'
    if not os.path.exists(pfile+'.py'):
        print('The prescription '+pfile+'.py does not exist.  Exiting.')
        sys.exit()
    pd = importlib.import_module(pfile)

    #model_param_set = convert_to_dict(param_names, parameter_set)
    model, model_directory = pd.write_run_mcfost(velax=velax, vsyst=6.04, ozstar=True, **model_param_set)
    model_lp = np.sum(model.lines[:, :, :], axis=(1, 2))

    return model_lp




