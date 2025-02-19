import sys, os, subprocess
import emcee
from .corner_plot import *
from .comparison_plot import *
import importlib


def make_plots(params='all', corner_plot=True, truths=None, to_plot='best', cs=False, eb3=False, disk='DM_Tau', reimage=False):

    directory = '.'
    for filename in os.listdir(directory):
        if filename.endswith('h5'):
            h5file = directory+'/'+filename

    directory += '/tracking_plots'

    if os.path.isdir(directory) == False:
        subprocess.call("mkdir -p "+ directory, shell = True)


    # Check number of steps to determine burnin/autocorrelation parameters

    reader = emcee.backends.HDFBackend(h5file, read_only=True)
    steps = reader.iteration
    autocorr = True
    Ntau = 20

    if steps <= 600:
        burnin = 0
        autocorr = True
    elif steps <= 1000:
        burnin = 500
        Ntau = 100
        autocorr = True
    elif steps <= 2500:
        burnin = 800
        Ntau = 100
    else:
        burnin = steps-2000
        Ntau = 300

    # Generate corner plot, traces, autocorrelation and print percentiles

    prec, average = corner_plot_updates(h5file, directory, params, burnin, autocorr, Ntau, corner_plot, truths, cs)

    if disk == 'DM_Tau':
        vsyst = 6.04
    elif disk == 'V4046':
        vsyst = 2.933
    else:
        print('Only set up for DM Tau and V4046')
        return

    # Find best likelihood model
    if to_plot == 'best':
        best_params, best_likelihood, iteration = find_best_params(h5file)
        print('Best params: ', best_params)
        print('Likelihood: ', best_likelihood)
        line_prof_image_plane(best_params, params, best_likelihood, iteration, directory, prec, eb3, disk=disk, vsyst=vsyst)
        if reimage:
            print('')

    else:
        print('Using median values')
        print(average)
        line_prof_image_plane(average, params, None, None, directory, prec, eb3, disk=disk, vsyst=vsyst)
        if reimage:
            print('')


    # nice trace plot for paper
    plot_traces_paper(h5file, params, cs)

    # Reimaged model
    # import data, fixed kwargs from run_mcmc file
    # data = '/home/chardima/runs/FITTING_EXPERIMENTS/concatenated.ms'
    # run_setup = importlib.import_module('run_mcmc_test')
    # fixed_kwargs = run_setup.fixed_kw
    # reimage_model(params=best_params, data=data, directory=directory,
    #               fixed_kwargs=fixed_kwargs, likelihood=best_likelihood,
    #               iteration=iteration)

def make_residual_cube(param_names):

    directory = '.'
    for filename in os.listdir(directory):
        if filename.endswith('h5'):
            h5file = directory+'/'+filename

    params, likelihood, iteration = find_best_params(h5file)

    cropped_cube = '/home/chardima/runs/DM_Tau_cutout_FOV_15.fits'
    data = casa.Cube(cropped_cube)
    beamarea = np.pi * data.bmaj * data.bmin / (4.0 * np.log(2.0))
    pixarea = data.pixelscale**2
    rms = 0.0008412888 # Jy/beam

    check_para_matches_data('csalt.para', data)

    pfile = 'parametric_disk_MCFOST'
    if not os.path.exists(pfile+'.py'):
        print('The prescription '+pfile+'.py does not exist.  Exiting.')
        sys.exit()
    pd = importlib.import_module(pfile) 

    params = convert_to_dict(param_names, params)

    velax = data.velocity*1000
    model, model_directory = pd.write_run_mcfost(velax=velax, vsyst=6.04, ozstar=False, **params)
    residuals = mcfost.Line(model_directory+'/data_CO')
    velax = data.velocity


    # Convolve model so we can subtract from data

    for i in range(len(model.lines)):
        model.plot_map(v=velax[i], bmaj=data.bmaj, bmin=data.bmin, bpa=data.bpa, per_beam=True)
        convolved = model.last_image
        residuals.lines[i] = np.array(data.image[i]) - np.array(convolved)

    from astropy.io import fits

    with fits.open(f'{model_directory}/data_CO/lines.fits.gz') as hdul:
        hdul[0].data = residuals.lines
        subprocess.call("mkdir -p data_CO_residuals", shell = True)
        hdul.writeto('data_CO_residuals/lines.fits.gz')

    return