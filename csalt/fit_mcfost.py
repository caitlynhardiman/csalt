# Import libraries
import os, sys
from csalt import *


# Class definition
class setup_fit():

    def __init__(self,
                 msfile=None,
                 code=None,
                 postfile='/DMTau.h5',
                 append: bool = False,
                 mpi: bool = False,
                 mode='iter',
                 param=None,
                 vra_fit=[4.06e3, 8.06e3],
                 vcensor=None,
                 nwalk=128,
                 ninits=300,
                 nsteps=5000,
                 nthreads=32,
                 nu_rest=345.796e9,
                 FOV=6.375,
                 Npix=256,
                 dist=144.5,
                 cfg_dict={}):
        
        if msfile is None:
            print('Need to give an ms file as input!')
            return
        
        if postfile is None:
            print('Please give a name for the backend')
            return
        
        # Model Setups
        if code is None:
            print('Using mcfost as default code for modelling')
            self.mtype = 'MCFOST'
        else:
            self.mtype = code
        self.mode = mode
        self.vra_fit = vra_fit
        self.vcensor = vcensor
        # I/O
        self.datafile = msfile
        self.post_dir = 'storage/posteriors/'+self.mtype
        self.postfile = postfile
        # Inference Setups
        self.nwalk = nwalk
        self.ninits = ninits
        self.nsteps = nsteps
        self.nthreads = nthreads
        self.append = append
        # Fixed Parameters
        self.nu_rest = nu_rest
        self.FOV = FOV
        self.Npix = Npix
        self.dist = dist
        self.cfg_dict = cfg_dict
        self.fixed = nu_rest, FOV, Npix, dist, cfg_dict
        self.mpi = mpi
        self.param = param

        if param==None:
            self.priorfile = self.mtype
        else:
            self.priorfile = self.mtype+'_'+self.param


        if self.mpi:
            from schwimmbad import MPIPool
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            print(rank)
            if rank == 0:
                self._setup_priors_and_postdir()
        else:
            self._setup_priors_and_postdir()


    def _setup_priors_and_postdir(self):
        """
        Sets up the prior functions and if necessary, makes the posteriors
        directory
        """
        if not os.path.exists('priors_'+self.priorfile+'.py'):
            print('There is no such file "priors_"+self.mtype+".py".\n')
            sys.exit()
        else:
            os.system('rm priors.py')
            os.system('cp priors_'+self.priorfile+'.py priors.py')
        if not os.path.exists(self.post_dir):
            os.system('mkdir -p '+self.post_dir)


    def mcmc_fit(self):
        """
        Performs mcmc fit of data visibilities with models as specified in setup
        No extra parameters required
        """
        
        run_emcee(self.datafile, self.fixed, code=self.mtype, vra=self.vra_fit,
                  vcensor=self.vcensor, nwalk=self.nwalk, ninits=self.ninits,
                  nsteps=self.nsteps, outfile=self.post_dir+self.postfile,
                  mode=self.mode, nthreads=self.nthreads, append=self.append,
                  mpi=self.mpi, param=self.param)

        sample_posteriors(self, msfile, vra=None, vcensor=None, kwargs=None,
                          restfreq=230.538e9, chbin=1, well_cond=300,
                          Nwalk=75, Ninits=20, Nsteps=1000, 
                          outpost='stdout.h5', append=False, Nthreads=6, param=None)
        

    def initialise(self):
        """
        Initialises the mcmc such that you could pass in a theta array and obtain a
        single log likelihood value etc.
        """
        
        ndim = len(pri_pars)
        print(ndim)
        data = fitdata(self.datafile, vra=self.vra_fit, nu_rest=self.nu_rest, chbin=3)
        p0 = init_priors(nwalk=self.nwalk, ndim=ndim)
        data = build_cache(p0, data, self.fixed, code=self.mtype, mode=self.mode, param = self.param)
        return data
    

    def get_probability(self, data, theta):
        """
        Returns the log probability of a specific theta array fitting the data
        Need to run initialise function first
        """

        set_globals(data, self.fixed)
        log_posterior, log_prior = lnprob(theta, code_=self.mtype)
        return log_posterior, log_prior
    

    def plot_visibilities(self, data, theta, mcube=None):
        """
        Plots the difference between the model visibilities and data visibilities
        by EB for given polarisation and channel
        Need to run initialise function first
        """
        plot(data, self.fixed, self.mtype, theta, mcube)

    def model_to_model(self, theta):
        """
        Runs a model to model fit so we can verify that the method is working
        """
        if self.mtype == 'MCFOST':
            from csalt.data_mcfost import fitdata
        data = fitdata(self.datafile, vra=self.vra_fit, nu_rest=self.nu_rest, chbin=3)
        p0 = [theta]
        data = build_cache(p0, data, self.fixed, code=self.mtype, mode=self.mode)

        model_vis = get_model_vis(theta, data, self.fixed, code_=self.mtype, mpi=self.mpi)

        run_emcee(self.datafile, self.fixed, code=self.mtype, vra=self.vra_fit,
                  vcensor=self.vcensor, nwalk=self.nwalk, ninits=self.ninits,
                  nsteps=self.nsteps, outfile=self.post_dir+self.postfile,
                  mode=self.mode, nthreads=self.nthreads, append=self.append,
                  mpi=self.mpi, model_vis=model_vis)
