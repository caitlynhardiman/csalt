import emcee
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import corner
from math import ceil


parameters_info = {
    'inclination': {'unit': r'$i$ ($^\circ$)', 'limits_DMTau': [25, 50]},
    'stellar_mass': {'unit': r'$M_{\rm star}$ (M$_\odot$)', 'limits_DMTau': [0.3, 0.9]},
    'scale_height': {'unit': r'$h_{0}$ (au)', 'limits_DMTau': [0, 30]},
    'r_c': {'unit': r'$R_{\rm c}$ (au)', 'limits_DMTau': [95, 520]},
    'flaring_exp': {'unit': r'$\Psi$', 'limits_DMTau': [1.2, 1.5]},
    'PA': {'unit': r'PA ($^{\circ}$)', 'limits_DMTau': [140, 170]},
    'dust_param': {'unit': r'$\rm log( \it \alpha_{\rm dust})$', 'limits_DMTau': [-5, -1]},
    'vturb': {'unit': r'$v_{\rm turb}$ (km s$^{-1}$)', 'limits_DMTau': [0, 0.25]},
    'vturb_cs': {'unit': r'$f_{\rm turb}$', 'limits_DMTau': [0.01, 0.5]},
    'gas_mass': {'unit': r'$M_{\rm gas}$ (M$\odot$)', 'limits_DMTau': [0.01, 0.08]},
    'gasdust_ratio': {'unit': r'$\rm log( \it M_{\rm gas}$/$M_{\rm dust})$', 'limits_DMTau': [1, 2.5]},
    'co_abundance': {'unit': r'$\rm log( \it X_{\rm CO}$)', 'limits_DMTau': [-6, -4]},
    'depletion_factor': {'unit': r'$\rm log( \it f \rm _{freeze-out})$', 'limits_DMTau': [-10, -1]},
    'depletion_factor_cs': {'unit': r'$\it f \rm _{freeze-out}$', 'limits_DMTau': [0, 1]},
    'gamma': {'unit': r'$\gamma$', 'limits_DMTau': [0, 2]},
    'gamma_exp': {'unit': r'$\gamma_{\rm exp}$', 'limits_DMTau': [0, 2]},
    'stellar_radius': {'unit': r'$R_{\rm star}$ (R$_\odot$)', 'limits_DMTau': [0.8, 2.6]},
    'teff': {'unit': r'$T_{\rm eff}$ (K)', 'limits_DMTau': [3500, 4500]},
    'chi_ISM': {'unit': r'$\chi_{\rm ISM}$', 'limits_DMTau': [0, 1]},
    'correct_Tgas': {'unit': r'$\rm T_{\rm gas}$', 'limits_DMTau': [0, 10]}
}

limits_cs = [[25, 50], [0.3, 0.9], [0, 30], [95, 520], [1.2, 1.5], [140, 170], 
          [-5, -1], [0, 0.25], [0.01, 0.08], [1, 2.5], [-6, -5.2], [0, 0.1],
          [0, 2], [0.8, 2.6]]

limits_cs_restricted = [[0.4, 0.6], [0, 20], [95, 1000], [1.1, 1.5], 
          [-5, -1], [0.01, 0.08], [1, 2.5], [-6, -4], [0, 2], [0.8, 2.6]]
          
limits_global = [[25, 50], [0.3, 0.9], [0, 30], [95, 520], [1.2, 1.5], [140, 170], 
          [-5, -1], [0, 0.25], [0.01, 0.08], [1, 2.5], [-6, -5.2], [-10, -1],
          [0, 2], [0.8, 2.6]]

limits_global_restricted = [[0.3, 0.9], [0, 30], [95, 520], [1.2, 1.5], 
          [-5, -1], [0, 0.25], [0.01, 0.08], [1, 2.5], [-6, -5.2], [0, 0.1],
          [0, 2], [0.8, 2.6]]



def corner_plot_updates(h5file, directory, params, burnin, autocorr, Ntau, corner_plot, truths, cs=False):
    
    reader = emcee.backends.HDFBackend(h5file, read_only=True)
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    samples_ = reader.get_chain(discard=burnin, flat=True)
    logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
    logprior_samples = reader.get_blobs(discard=burnin, flat=False)
    nsteps, nwalk, ndim = samples.shape
    print("Number of iterations: ", reader.iteration)

    lbls, units, limits = labels(params, cs)

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
        plot_corner(samples, ndim, units, truths, directory)

    return prec, average_values



###############################################################################################################

def labels(params, cs=False):
    if params == 'all':
        lbls = ['inc', 'M', 'h', 'rc', 'psi', 'PA', 'dust α', 'vturb', 'gas mass', 'g/d mass ratio']
    elif params == 'extra':
        lbls = ['inc', 'M', 'h', 'rc', 'psi', 'PA', 'dust α', 'vturb', 'gas mass', 'g/d mass ratio', 'teff', 'mu_RA', 'mu_DEC']
    elif params == 'co_df':
        lbls = ['inc', 'M', 'h', 'rc', 'psi', 'PA', 'dust α', 'vturb', 'gas mass', 'g/d mass ratio', 'co abundance', 'dep. factor']
    elif params == 'gammas':
        lbls = ['inc', 'M', 'h', 'rc', 'psi', 'PA', 'dust α', 'vturb', 'gas mass', 'g/d mass ratio', 'co abundance', 'dep. factor', 'gamma', 'gamma_exp']
    else:
        lbls = params
    
    units = []
    limits = []
    for param in lbls:
        if param == 'vturb' and cs:
            units.append(parameters_info['vturb_cs']['unit'])
            limits.append(parameters_info['vturb_cs']['limits_DMTau'])
        elif param == 'depletion_factor' and cs:
            units.append(parameters_info['depletion_factor_cs']['unit'])
            limits.append(parameters_info['depletion_factor_cs']['limits_DMTau'])
        else:
            units.append(parameters_info[param]['unit'])
            limits.append(parameters_info[param]['limits_DMTau'])

    return lbls, units, limits

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
            tau_ix[i] = np.nanmean(tau)

        fig = plt.figure()
        plt.plot(ix, tau_ix, '-o')
        plt.xlabel('steps')
        plt.ylabel('autocorr time (steps)')
        plt.xlim([0, Nmax])

        plt.ylim(0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min()))
        fig.savefig(directory+'/autocorr.pdf')
        fig.clf()

def plot_traces(logpost_samples, logprior_samples, nwalk, nsteps, ndim, 
                samples, truths, lbls, directory):
    
    default_color='k'

    # Plot the traces
    nrows = ceil((ndim+2)/6)
    print(nrows)
    fig = plt.figure(figsize=(20, 3*nrows))
    gs = gridspec.GridSpec(nrows, 6)

    # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        color = default_color
        alpha = 0.03
        ax.plot(np.arange(nsteps),
                logpost_samples[:,iw] - logprior_samples[:,iw],
                color=color, alpha=alpha)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    #ax.set_ylabel('log likelihood', fontsize=6)
    ax.set_xticklabels([])
    ax.text(0.95, 0.05, 'ln L', fontsize=12, ha='right', color='purple',
            transform=ax.transAxes)

    # log-prior
    ax = fig.add_subplot(gs[0,1])
    for iw in range(nwalk):
        color = default_color
        alpha = 0.03
        ax.plot(np.arange(nsteps), logprior_samples[:, iw],
                color=color, alpha=alpha)
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
            color = default_color
            alpha=0.03
            ax.plot(np.arange(nsteps), samples[:, iw, idim],
                    color=color, alpha=alpha)
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
    fig.savefig(directory+'/traces.pdf')
    fig.clf()

def plot_corner(samples, ndim, lbls, truths, directory):

    levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
    flat_chain = samples.reshape(-1, ndim)

    fig = corner.corner(flat_chain, labels=lbls,
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                        bins=20, label_kwargs=dict(size=20, font='serif'), title_quantiles = [0.1,0.5,0.9],
                        quantiles = [0.1,0.5,0.9], color='purple', hist_kwargs=dict(density=True, lw=1),
                        plot_density=False, plot_datapoints=True, fill_contours=True,  max_n_ticks=2, truths=truths, labelpad=0.3)
    
    for ax in fig.get_axes():  
      ax.set_xlabel(ax.get_xlabel(), size=20)
      ax.set_ylabel(ax.get_ylabel(), size=20)
      ax.tick_params(labelsize=18)
      ax.set_title(ax.get_title(), size=20)
    
    
    fig.savefig(directory+'/corner.pdf', dpi=300, bbox_inches='tight')
    fig.clf()

    print(flat_chain.shape)


    # plt.figure(dpi=200, figsize=[4,4])
    # plt.scatter(flat_chain[:,1], flat_chain[:,7]*1e3, c="k", alpha=0.2, s=0.1)
    # plt.xlabel("M_star [M_sun]")
    # plt.ylabel("v_turb [m/s]")
    # plt.xlim(0.48, 0.6)
    # plt.ylim(0.048e3, 0.065e3)
    # plt.savefig(directory+"/scatter_mstar_vturb.pdf", bbox_inches="tight")

def precision(params):

    all_params_prec = [0.1, 0.001, 0.1, 0.1, 0.0001, 0.1, 0.000001, 0.001, 0.000001, 
                       0.001, 0.1, 0.01, 0.01, 1e-6, 0.001, 0.001, 0.1, 0.1, 0.01, 0.1, 0.001, 0.1, 0.01, 0.01]
    param_names = ['inclination', 'stellar_mass', 'scale_height', 'r_c', 
                   'flaring_exp', 'PA', 'dust_param', 'vturb', 'gas_mass', 'gasdust_ratio', 'teff', 'mu_RA', 'mu_DEC', 
                   'co_abundance', 'depletion_factor', 'gamma', 'gamma_exp', 'ism_chi', 'stellar_radius', 'tgas', 'vsyst', 'teff', 'chi_ISM', 'correct_Tgas']

    if params == 'all':
        prec = all_params_prec[:10]
    elif params == 'extra':
        prec = all_params_prec[:13]
    elif params == 'co_df':
        prec = all_params_prec[:10]+all_params_prec[-4:-2]
    elif params == 'gammas':
        prec = all_params_prec[:10]+all_params_prec[-4:]
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


def plot_traces_paper(h5file, params, cs=False, restricted=False):

    reader = emcee.backends.HDFBackend(h5file, read_only=True)
    samples = reader.get_chain(discard=0, flat=False)
    logpost_samples = reader.get_log_prob(discard=0, flat=False)
    logprior_samples = reader.get_blobs(discard=0, flat=False)
    nsteps, nwalk, ndim = samples.shape
    print("Number of iterations: ", reader.iteration)

    lbls, units, limits = labels(params, cs)
    
    default_color='purple'
    # if not restricted:
    #     if cs:
    #         limits = limits_cs
    #     else:
    #         limits = limits_global
    # else:
    #     if cs:
    #         limits = limits_cs_restricted
    #     else:
    #         limits = limits_global_restricted

    # Plot the traces
    nrows = ceil((ndim+1)/5)
    fig = plt.figure(figsize=(25, 12))
    gs = gridspec.GridSpec(nrows, 5)

     # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        color = default_color
        alpha = 0.03
        ax.plot(np.arange(nsteps),
                logpost_samples[:,iw] - logprior_samples[:,iw],
                color=color, alpha=alpha)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=20)
    ax.set_ylabel('log likelihood', fontsize=20)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # cycle through parameters
    ax_ixl = [np.floor_divide(idim, 5) for idim in np.arange(1, ndim+1)]
    ax_ixh = [(idim % 5) for idim in np.arange(1, ndim+1)]
    print(ax_ixl, ax_ixh)
    for idim in range(ndim):
        ax = fig.add_subplot(gs[ax_ixl[idim], ax_ixh[idim]])
        for iw in range(nwalk):
            color = default_color
            alpha=0.03
            ax.plot(np.arange(nsteps), samples[:, iw, idim],
                    color=color, alpha=alpha)
        ax.set_xlim([0, nsteps])
        ax.set_ylim(limits[idim])
        ax.tick_params(which='both', labelsize=20)
        if idim == 11:
            ax.set_ylabel(units[idim], fontsize=20)
        else:
            ax.set_ylabel(units[idim], fontsize=25)
        if idim==9:
            ax.set_xlabel('steps', fontsize=20)
        if idim == 10:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.2)
    fig.savefig('traces_testing.pdf', bbox_inches='tight', pad_inches=0.2)
    fig.clf()



def plot_corner_small(samples, ndim, lbls, directory):

    flat_chain = samples.reshape(-1, ndim)
    # inclination = 0, stellar mass = 1, vturb = 7
    my_params_chain = flat_chain[:, [0, 1, 7]]    
    fig = corner.corner(my_params_chain, labels=lbls,
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                        bins=20, label_kwargs=dict(size=14, font='serif'), #show_titles=True, title_quantiles = [0.1,0.5,0.9],
                        quantiles = [0.1,0.5,0.9], color='purple', hist_kwargs=dict(density=True, lw=1),
                        plot_density=False, plot_datapoints=True, fill_contours=True,  max_n_ticks=3)
    
    for ax in fig.get_axes():  
      ax.set_xlabel(ax.get_xlabel(), size=17)
      ax.set_ylabel(ax.get_ylabel(), size=17)
                        
    fig.savefig(directory+'/minicorner.pdf', dpi=300)
    fig.clf()