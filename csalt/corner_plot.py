import emcee
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import corner


def corner_plot_updates(h5file, directory, params, burnin, autocorr, Ntau, corner_plot, truths):
    
    reader = emcee.backends.HDFBackend(h5file, read_only=True)
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    samples_ = reader.get_chain(discard=burnin, flat=True)
    logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
    logprior_samples = reader.get_blobs(discard=burnin, flat=False)
    nsteps, nwalk, ndim = samples.shape
    print("Number of iterations: ", reader.iteration)

    lbls = labels(params)

    prec = precision(params)
    average_values = []

    for idim in range(ndim):
        fmt = '{:.'+str(np.abs(int(np.log10(prec[idim]))))+'f}'
        pk, hi, lo, med = post_summary(samples_[:,idim], prec=prec[idim], mu='not_peak')
        average_values.append(pk)
        print((lbls[idim] + ' = '+fmt+' +'+fmt+' / -'+fmt).format(pk, hi, lo))
    print(' ')

    if autocorr:
        plot_autocorr(all_samples, Ntau, directory)

    plot_traces(logpost_samples, logprior_samples, nwalk, nsteps, ndim, 
                samples, truths, lbls, directory)

    if corner_plot:
        plot_corner(samples, ndim, lbls, truths, directory)

    return prec, average_values



###############################################################################################################

def labels(params):
    if params == 'all':
        lbls = ['inc', 'M', 'h', 'rc', 'psi', 'PA', 'dust α', 'vturb', 'gas mass', 'g/d mass ratio']
    else:
        lbls = params
    return lbls

def plot_autocorr(all_samples, Ntau, directory):

    Nmax = all_samples.shape[0]
    if (Nmax > Ntau):
        tau_ix = np.empty(int(Nmax / Ntau))
        ix = np.empty(int(Nmax / Ntau))
        for i in range(len(tau_ix)):
            nn = (i + 1) * Ntau
            ix[i] = nn
            tau = emcee.autocorr.integrated_time(all_samples[:nn,:,:],
                                                     tol=0)
            tau_ix[i] = np.mean(tau)

    fig = plt.figure()
    plt.plot(ix, tau_ix, '-o')
    plt.xlabel('steps')
    plt.ylabel('autocorr time (steps)')
    plt.xlim([0, Nmax])
    plt.ylim([0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min())])
    fig.savefig(directory+'/autocorr.png')
    fig.clf()

def plot_traces(logpost_samples, logprior_samples, nwalk, nsteps, ndim, 
                samples, truths, lbls, directory):

    # Plot the traces
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 6)

    # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps),
                logpost_samples[:,iw] - logprior_samples[:,iw],
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    #ax.set_ylabel('log likelihood', fontsize=6)
    ax.set_xticklabels([])
    ax.text(0.95, 0.05, 'ln L', fontsize=12, ha='right', color='purple',
            transform=ax.transAxes)

    # log-prior
    ax = fig.add_subplot(gs[0,1])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps), logprior_samples[:, iw],
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.set_ylim([np.min(logprior_samples[:, iw]) - 0.05,
                    np.max(logprior_samples[:, iw]) + 0.05])
    ax.tick_params(which='both', labelsize=6)
    #ax.set_ylabel('log prior', fontsize=6)
    ax.set_xticklabels([])
    ax.text(0.95, 0.05, 'ln prior', fontsize=12, ha='right', color='purple',
                transform=ax.transAxes)

    # now cycle through parameters
    ax_ixl = [np.floor_divide(idim, 6) for idim in np.arange(2, ndim+2)]
    ax_ixh = [(idim % 6) for idim in np.arange(2, ndim+2)]
    for idim in range(ndim):
        ax = fig.add_subplot(gs[ax_ixl[idim], ax_ixh[idim]])
        for iw in range(nwalk):
            ax.plot(np.arange(nsteps), samples[:, iw, idim],
                    color='k', alpha=0.03)
        if truths is not None:
            ax.plot([0, nsteps], [truths[idim], truths[idim]], '--C1', lw=1.5)
        ax.set_xlim([0, nsteps])
        ax.tick_params(which='both', labelsize=6)
        #ax.set_ylabel(lbls[idim], fontsize=6)
        if idim != 10:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('steps', fontsize=6)
        ax.text(0.95, 0.05, lbls[idim], fontsize=12, ha='right', color='purple',
                    transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.20, hspace=0.05)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(directory+'/traces.png')
    fig.clf()

def plot_corner(samples, ndim, lbls, truths, directory):

    levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
    flat_chain = samples.reshape(-1, ndim)
    fig = corner.corner(flat_chain, plot_datapoints=True, levels=levs,
                            labels=lbls, truths=truths)
                            #, color='white', smooth=0.9, no_fill_contours=True)
    fig.savefig(directory+'/corner.png', dpi=300)
    fig.clf()

def precision(params):

    all_params_prec = [0.1, 0.001, 0.1, 0.1, 0.0001, 0.1, 0.000001, 0.001, 0.000001, 0.1]
    param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 
                   'flaring_exp', 'PA', 'dust_param', 'vturb', 'gas_mass', 'gd_ratio']

    if params == 'all':
        prec = all_params_prec
    else:
        prec = []
        for param in params:
            index = -1
            for i, parameter in enumerate(param_names):
                if param == parameter:
                    prec.append(all_params_prec[i])
                    index = i
                    break
            if index == -1:
                print(param+' not a valid parameter name!')
                return

    return prec       

def post_summary(p, prec=0.1, mu='peak', CIlevs=[84.135, 15.865, 50.]):

    # calculate percentiles as designated
    CI_p = np.percentile(p, CIlevs)

    # find peak of posterior
    if (mu == 'peak'):
        kde_p = stats.gaussian_kde(p)
        ndisc = int(np.round((CI_p[0] - CI_p[1]) / prec))
        x_p = np.linspace(CI_p[1], CI_p[0], ndisc)
        pk_p = x_p[np.argmax(kde_p.evaluate(x_p))]
    else:
        pk_p = np.percentile(p, 50.)

    # return the peak and upper, lower 1-sigma
    return (pk_p, CI_p[0]-pk_p, pk_p-CI_p[1], CI_p[2])