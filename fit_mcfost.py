# Import libraries
import os, sys
from csalt import *
from schwimmbad import MPIPool

# Class definition

class setup_fit():

    def __init__(self,
                 h5file: None,
                 postfile='DMTau.h5',
                 append: bool = False,
                 code: None,
                 mpi: bool = False,
                 mode='iter',
                 vra_fit=[4e3, 8e3],
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
        
        if h5file is None:
            print('Need to give a h5 file as input!')
            return
            
        if postfile is None:
            print('Please give a name for the backend')
            return
        
        # Model Setups
        if code is None:
            print('Using mcfost as default code for modelling')
            self.mtype = 'mcfost'
        else:
            self.mtype = code
        self.mode = mode
        self.vra_fit = vra_fit
        self.vcensor = vcensor
        
        # I/O
        self.datafile = h5file
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
        
        if self.mpi:
            os.environ["OMP_NUM_THREADS"] = "1"
            with MPIPool() as pool:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                self._setup_priors_and_postdir()
        else:
            self._setup_priors_and_postdir()
        
        

    def _setup_priors_and_postdir(self):
    
        """
        Sets up the prior functions and if necessary, makes the posteriors
        directory
        """
        
        if not os.path.exists('priors_'+self.mtype+'.py'):
            print('There is no such file "priors_"+self.mtype+".py".\n')
            sys.exit()
        else:
            os.system('rm priors.py')
            os.system('cp priors_'+self.mtype+'.py priors.py')

        if not os.path.exists(self.post_dir):
            os.system('mkdir '+self.post_dir)

    
    def mcmc_fit(self):
    
        """
        Performs mcmc fit of data visibilities with models as specified in setup
        No extra parameters required
        """
    
        run_emcee(self.datafile, self.fixed, code=self.mtype, vra=self.vra_fit,
                  vcensor=self.vcensor, nwalk=self.nwalk, ninits=self.ninits,
                  nsteps=self.nsteps, outfile=self.post_dir+self.postfile,
                  mode=self.mode, nthreads=self.nthreads, append=self.append,
                  mpi=self.mpi)
                  

    def initialise(self):
    
        """
        Initialises the mcmc such that you could pass in a theta array and obtain a
        single log likelihood value etc.
        """
        
        data = fitdata(self.datafile, vra=self.vra_fit, nu_rest=self.nu_rest, chbin=2)
        p0 = init_priors(nwalk=self.nwalk)
        data = build_cache(p0, data, self.fixed, code=self.mtype, mode=self.mode)
        return data
        
    def get_probability(self, data, theta):
    
        """
        Returns the log probability of a specific theta array fitting the data
        Need to run initialise function first
        """
        
        global data_
        data_ = data
        global fixed_
        fixed_ = fixed
        
        log_posterior, log_prior = lnprob(theta)
        return log_posterior, log_prior
    
    
    def plot(self, data, theta, mcube=None):
    
        """
        Plots the difference between the model visibilities and data visibilities
        by EB for given polarisation and channel
        Need to run initialise function first
        """

       plot(data, self.fixed, self.mtype, theta, mcube)
