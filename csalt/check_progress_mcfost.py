import sys, os, subprocess
import emcee
from .corner_plot import *
from .comparison_plot import *


def make_plots(params='all', corner_plot=True, truths=None):

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
    Ntau = 200

    if steps <= 300:
        burnin = 0
        autocorr = False
    elif steps <= 700:
        burnin = 200
    else:
        burnin = 600

    # Generate corner plot, traces, autocorrelation and print percentliles

    prec, average = corner_plot_updates(h5file, directory, params, burnin, autocorr, Ntau, corner_plot, truths)
    print(average)

    # Find best likelihood model
    best_params, best_likelihood, iteration = find_best_params(h5file)
    print('Best params: ', best_params)
    print('Likelihood: ', best_likelihood)


    # Line profile and image plane best model
    line_prof_image_plane(best_params, params, best_likelihood, iteration, directory, prec)

    # Reimaged model
    #reimage_model(best_params, best_likelihood, iteration, directory)