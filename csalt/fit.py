import os, sys, time, importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from csalt.data import *
from csalt.data_mcfost import fitdata as fitdata_mcfost
from csalt.models import *
from priors import *
import emcee
import corner
from scipy import stats
import scipy.constants as sc
from multiprocessing import Pool

# data_ = None
# fixed_ = None


# log-posterior calculator
def lnprob(theta, code_='default', mpi=False, model_vis=None, param=None):

    # compute the log-prior and return if problematic
    lnT = np.sum(logprior(theta)) * data_['nobs']
    if lnT == -np.inf:
        return -np.inf, -np.inf

    # loop through observations to compute the log-likelihood
    lnL = 0
    mcube = None
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        if EB == 0 or mcube is None:
            mvis, mcube = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)], code=code_, mpi=mpi, param=param)
        else:
            mvis, mcube = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)], code=code_,
                             mcube=mcube, mpi=mpi, param=param)

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin,
                                          dat.nvis)), weights=wt, axis=2)


        # compute the residuals (stack both pols)
        if model_vis is None:
            resid = np.hstack(np.absolute(dat.vis - mvis_b))
        else:
            resid = np.hstack(np.absolute(model_vis[str(EB)] - mvis_b))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL += -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior
    return lnL + dat.lnL0 + lnT, lnT



# log-posterior calculator
def lnprob_naif(theta):

    # compute the log-prior and return if problematic
    lnT = np.sum(logprior(theta)) * data_['nobs']
    if lnT == -np.inf:
        return -np.inf, -np.inf

    # loop through observations to compute the log-likelihood
    lnL = 0
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        mvis = vismodel_naif(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)])

        # compute the residuals (stack both pols)
        resid = np.hstack(np.absolute(dat.vis - mvis))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL += -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior
    return lnL + dat.lnL0 + lnT, lnT


def lnprob_naif_wdoppcorr(theta):

    # compute the log-prior and return if problematic
    lnT = np.sum(logprior(theta)) * data_['nobs']
    if lnT == -np.inf:
        return -np.inf, -np.inf

    # loop through observations to compute the log-likelihood
    lnL = 0
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        mvis = vismodel_naif_wdoppcorr(theta, fixed_, dat,
                                       data_['gcf'+str(EB)],
                                       data_['corr'+str(EB)])

        # compute the residuals (stack both pols)
        resid = np.hstack(np.absolute(dat.vis - mvis))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL += -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior
    return lnL + dat.lnL0 + lnT, lnT


def run_sampler(mode, nwalk, ndim, steps, pos, backend, code, pool, mpi, model_vis, param):
    if mode == 'naif':
        print('\n Note: running in naif mode... \n')
        sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob_naif,
                                        pool=pool, backend=backend, kwargs={"code_": code})
        t0 = time.time()
        sampler.run_mcmc(pos, steps, progress=True)
        t1 = time.time()
    elif mode == 'naif_wdoppcorr':
        print('\n Note: running in naif mode with doppler correction... \n')
        sampler = emcee.EnsembleSampler(nwalk, ndim,
                                        lnprob_naif_wdoppcorr,
                                        pool=pool, backend=backend,
                                        kwargs={"code_": code})
        t0 = time.time()
        sampler.run_mcmc(pos, steps, progress=True)
        t1 = time.time()
    else:
        sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                        backend=backend, kwargs={"code_": code, "mpi": mpi, "model_vis": model_vis, 'param': param})
        t0 = time.time()
        print('Running MCMC')
        sampler.run_mcmc(pos, steps, progress=True)
        t1 = time.time()

    return sampler, t0, t1
   

def run_emcee(datafile, fixed, code=None, vra=None, vcensor=None,
              nwalk=75, ninits=200, nsteps=1000, chbin=3,
              outfile='stdout.h5', append=False, mode='iter', nthreads=6,
              mpi=False, model_vis=None, param=None):


    # load the data
    global data_
    print('Reading in data from h5 file')
    if code=='MCFOST' or "MCFOST_SMALL":
        data_ = fitdata_mcfost(datafile, vra=vra, vcensor=vcensor,
                    nu_rest=fixed[0], chbin=chbin)
    else:
        data_ = fitdata(datafile, vra=vra, vcensor=vcensor,
                    nu_rest=fixed[0], chbin=chbin)

    print('Data read in from h5 file')

    # assign fixed
    global fixed_
    fixed_ = fixed

    #set_globals(data_, fixed_)

    # initialize parameters using random draws from the priors
    ndim = len(pri_pars)
    p0 = init_priors(ndim, nwalk)

    print(ndim, ' Priors initialised')

    data_ = build_cache(p0, data_, fixed_, code=code, mode=mode, param=param)

    print('Preliminary models run')

    # mpi pool or multiprocess pool
    pool = None
    if mpi:
        from schwimmbad import MPIPool
        from mpi4py import MPI
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        print(nthreads)
        pool = Pool(processes=nthreads)

    # Configure backend for recording posterior samples
    post_file = outfile
    if not append:
        # run to initialize

        isampler, t0, t1 = run_sampler(mode, nwalk, ndim, ninits, p0, backend=None,
                                       code=code, pool=pool, mpi=mpi, model_vis=model_vis, param=param)

        print('Backend configured in %.2f hours' % ((t1 - t0) / 3600))

        # reset initialization to more compact distributions
        # this does random, uniform draws from the inner quartiles of the
        # walker distributions at the end initialization step (effectively
	    # pruning outlier walkers stuck far from the mode)
        isamples = isampler.get_chain()	  # [ninits, nwalk, ndim]-shaped
        lop0 = np.quantile(isamples[-1, :, :], 0.25, axis=0)
        hip0 = np.quantile(isamples[-1, :, :], 0.75, axis=0)
        p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]
        print('\nChains now properly initialized...\n')

        # prepare the backend file for the full run
        os.system('rm -rf '+post_file)
        backend = emcee.backends.HDFBackend(post_file)
        backend.reset(nwalk, ndim)

        if mpi:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # run the MCMC
        sampler, t0, t1 = run_sampler(mode, nwalk, ndim, nsteps, p00, backend, code, pool, mpi, model_vis, param)

    else:
        new_backend = emcee.backends.HDFBackend(post_file)
        print("Initial size: {0}".format(new_backend.iteration))

        new_sampler, t0, t1 = run_sampler(mode, nwalk, ndim, nsteps-new_backend.iteration, 
                                          pos=None, backend=new_backend, code=code, pool=pool, mpi=mpi, model_vis=model_vis, param=param)
        print("Final size: {0}".format(new_backend.iteration))

    print(' ')
    print(' ')
    print('This run took %.2f hours' % ((t1 - t0) / 3600))

    pool.close()

    return

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



def post_analysis(outfile, burnin=0, autocorr=False, Ntau=200,
                  corner_plot=True, truths=None):

    # load the emcee backend file
    reader = emcee.backends.HDFBackend(outfile)

    # parse the samples
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    samples_ = reader.get_chain(discard=burnin, flat=True)
    logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
    logprior_samples = reader.get_blobs(discard=burnin, flat=False)
    nsteps, nwalk, ndim = samples.shape

    # set parameter labels, truths (NOT HARDCODE!)
    lbls = ['incl', 'PA', 'M', 'r_l', 'z0', 'psi', 'Tb0', 'q', 'Tback', 'dV0',
            'tau0', 'p', 'vsys', 'dx', 'dy']


    # Plot the integrated autocorrelation time every Ntau steps
    if autocorr:
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
        fig.savefig('autocorr.png')
        fig.clf()


    # Plot the traces
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(3, 6)

    # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps),
                logpost_samples[:,iw] - logprior_samples[:,iw],
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log likelihood', fontsize=6)
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
    ax.set_ylabel('log prior', fontsize=6)
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
        ax.set_ylabel(lbls[idim], fontsize=6)
        if idim != 10:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('steps', fontsize=6)
        ax.text(0.95, 0.05, lbls[idim], fontsize=12, ha='right', color='purple',
                transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.20, hspace=0.05)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.99)
    fig.savefig('traces.png')
    fig.clf()


    # corner plot
    if corner_plot:
        levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
        flat_chain = samples.reshape(-1, ndim)
        fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs,
                            labels=lbls, truths=truths)
        fig.savefig('corner.png')
        fig.clf()


    # Parameter inferences (1-D marginalized)
    print(' ')
    prec = [0.01, 0.01, 0.001, 0.1, 0.0001, 0.01, 0.1, 0.001, 0.1, 0.1, 0.01,
            0.01, 0.1, 0.0001, 0.0001]
    for idim in range(ndim):
        fmt = '{:.'+str(np.abs(int(np.log10(prec[idim]))))+'f}'
        pk, hi, lo, med = post_summary(samples_[:,idim], prec=prec[idim])
        print((lbls[idim] + ' = '+fmt+' +'+fmt+' / -'+fmt).format(pk, hi, lo))
    print(' ')

def post_analysis_mcfost(outfile, burnin=0, autocorr=False, Ntau=200,
                  corner_plot=True, truths=None):

    # load the emcee backend file
    reader = emcee.backends.HDFBackend(outfile)

    # parse the samples
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    samples_ = reader.get_chain(discard=burnin, flat=True)
    logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
    logprior_samples = reader.get_blobs(discard=burnin, flat=False)
    nsteps, nwalk, ndim = samples.shape

    # set parameter labels, truths (NOT HARDCODE!)
    lbls = ['incl', 'M', 'h', 'rc', 'rin', 'psi', 'PA', 'dust a', 'vturb', 'dust mass', 'g/d mass ratio']
    poster_lbls = ['stellar mass (Msun)', 'vturb (km/s)']


    # Plot the integrated autocorrelation time every Ntau steps
    if autocorr:
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
        fig.savefig('autocorr.png')
        fig.clf()


    # Plot the traces
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(3, 6)

    # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps),
                logpost_samples[:,iw] - logprior_samples[:,iw],
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log likelihood', fontsize=6)
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
    ax.set_ylabel('log prior', fontsize=6)
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
        ax.set_ylabel(lbls[idim], fontsize=6)
        if idim != 10:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('steps', fontsize=6)
        ax.text(0.95, 0.05, lbls[idim], fontsize=12, ha='right', color='purple',
                transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.20, hspace=0.05)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.99)
    fig.savefig('traces.png')
    fig.clf()


    # corner plot
    if corner_plot:
        #plt.style.use('dark_background')
        levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
        flat_chain = samples.reshape(-1, ndim)
        # poster_chain = []
        # poster_vars = [1, 8]
        # for step in flat_chain:
        #     newstep = []
        #     for index in poster_vars:
        #         newstep.append(step[index])
        #     poster_chain.append(newstep)
        # poster_chain = np.array(poster_chain)
        fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs,
                            labels=lbls, truths=truths)
                            #, color='white', smooth=0.9, no_fill_contours=True)
        fig.savefig('corner.png', dpi=300)
        fig.clf()


    # Parameter inferences (1-D marginalized)
    print(' ')
    prec = [0.1, 0.01, 0.1, 0.1, 0.1, 0.001, 0.1, 0.0001, 0.001, 0.000001, 0.1]
    for idim in range(ndim):
        fmt = '{:.'+str(np.abs(int(np.log10(prec[idim]))))+'f}'
        pk, hi, lo, med = post_summary(samples_[:,idim], prec=prec[idim])
        print((lbls[idim] + ' = '+fmt+' +'+fmt+' / -'+fmt).format(pk, hi, lo))
    print(' ')

def post_analysis_mcfost_small(outfile, burnin=0, autocorr=False, Ntau=200,
                  corner_plot=True, truths=None):

    # load the emcee backend file
    reader = emcee.backends.HDFBackend(outfile)

    # parse the samples
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    samples_ = reader.get_chain(discard=burnin, flat=True)
    logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
    logprior_samples = reader.get_blobs(discard=burnin, flat=False)
    nsteps, nwalk, ndim = samples.shape

    # set parameter labels, truths (NOT HARDCODE!)
    lbls = ['M','vturb']


    # Plot the integrated autocorrelation time every Ntau steps
    if autocorr:
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
        fig.savefig('autocorr.png')
        fig.clf()


    # Plot the traces
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(3, 6)

    # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps),
                logpost_samples[:,iw] - logprior_samples[:,iw],
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log likelihood', fontsize=6)
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
    ax.set_ylabel('log prior', fontsize=6)
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
        ax.set_ylabel(lbls[idim], fontsize=6)
        if idim != 10:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('steps', fontsize=6)
        ax.text(0.95, 0.05, lbls[idim], fontsize=12, ha='right', color='purple',
                transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.20, hspace=0.05)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.99)
    fig.savefig('traces.png')
    fig.clf()


    # corner plot
    if corner_plot:
        levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
        flat_chain = samples.reshape(-1, ndim)
        fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs,
                            labels=lbls, truths=truths)
        fig.savefig('corner.png')
        fig.clf()


    # Parameter inferences (1-D marginalized)
    print(' ')
    prec = [0.01, 0.001]
    for idim in range(ndim):
        fmt = '{:.'+str(np.abs(int(np.log10(prec[idim]))))+'f}'
        pk, hi, lo, med = post_summary(samples_[:,idim], prec=prec[idim])
        print((lbls[idim] + ' = '+fmt+' +'+fmt+' / -'+fmt).format(pk, hi, lo))
    print(' ')

def build_cache(p0, data_, fixed_, code, mode=None, param=None):

    mcube = None
    for EB in range(data_['nobs']):
        #print('lnL0 = ', data_['lnL0_'+str(EB)]) 

        # initial model calculations
        if mode == 'naif':
            _mvis, gcf, corr = vismodel_naif(p0[0], fixed_, data_[str(EB)],
                                             return_holders=True)
        elif mode == 'naif_wdoppcorr':
            _mvis, gcf, corr = vismodel_naif_wdoppcorr(p0[0], fixed_,
                                                       data_[str(EB)],
                                                       return_holders=True)
        else:
            if EB==0 or mcube is not None:
                _mvis, gcf, corr, mcube = vismodel_def(p0[0], fixed_, data_[str(EB)],
                                            mtype=code, return_holders=True, mcube=mcube, param=param)
            else:
                _mvis, gcf, corr, mcube = vismodel_def(p0[0], fixed_, data_[str(EB)],
                                            mtype=code, return_holders=True, param=param)

        # add gcf, corr caches into data dictionary, indexed by EB
        data_['gcf'+str(EB)] = gcf
        data_['corr'+str(EB)] = corr

    return data_

def get_model_vis(theta, data_, fixed_, code_, mpi):
    mcube = None
    
    modelvis = {}
    
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        if EB == 0 or mcube is None:
            mvis, mcube = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)], code=code_, mpi=mpi)
        else:
            mvis, mcube = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)], code=code_,
                             mcube=mcube, mpi=mpi)

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin,
                                          dat.nvis)), weights=wt, axis=2)
        
        modelvis[str(EB)] = mvis_b
    
    return modelvis


def init_priors(ndim, nwalk):
    p0 = np.empty((nwalk, ndim))
    for ix in range(ndim):
        if pri_types[ix] == "normal" or pri_types[ix] == "uniform":
            _ = [str(pri_pars[ix][ip])+', ' for ip in range(len(pri_pars[ix]))]
            cmd = 'np.random.'+pri_types[ix]+'('+"".join(_)+str(nwalk)+')'
            p0[:,ix] = eval(cmd)
        elif pri_types[ix] == "truncnorm" or pri_types[ix] == "loguniform":
            if pri_types[ix] == "truncnorm":
                params = pri_pars[ix]
                mod_pri_pars = [(params[2]-params[0])/params[1], (params[3]-params[0])/params[1], params[0], params[1]]
                _ = [str(mod_pri_pars[ip])+', ' for ip in range(len(mod_pri_pars))]
            else:
                _ = [str(pri_pars[ix][ip])+', ' for ip in range(len(pri_pars[ix]))]
            cmd = 'stats.'+pri_types[ix]+'.rvs('+"".join(_)+'size='+str(nwalk)+')'
            p0[:,ix] = eval(cmd)
        else:
            raise NameError('Prior type unaccounted for')
    return p0

def set_globals(data, fixed):
    global data_
    global fixed_
    data_ = data
    fixed_ = fixed