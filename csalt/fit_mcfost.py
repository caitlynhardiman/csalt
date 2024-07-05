# Import libraries
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csalt.model import *
from csalt.helpers import *
import multiprocessing
from functools import partial
import pymcfost as mcfost
import casa_cube as casa

# Class definition
class setup_fit():

    def __init__(self,
                 msfile=None,
                 append: bool = False,
                 param=None,
                 vra_fit=[4.06e3, 8.06e3],
                 vspacing =None,
                 vcensor=None,
                 nwalk=128,
                 ninits=300,
                 nsteps=5000,
                 nthreads=32,
                 nu_rest=345.796e9,
                 FOV=6.375,
                 Npix=256,
                 dist=144.5,
                 image_plane=False,
                 data_cube=None, 
                 casa_sim=False,
                 cfg_dict={}):
        
        if msfile is None:
            print('Need to give an ms file as input!')
            return
        
        # code is mcfost
        self.mtype = 'MCFOST'
        
        self.vra_fit = vra_fit
        self.vcensor = vcensor
        self.vspacing = vspacing
        # I/O
        self.datafile = msfile
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

        if param is not None:
            print('Using param!')
            self.param = param
            self.priors_prescription = self.mtype
            for parameter in self.param:
                self.priors_prescription += '_'+parameter
        else:
            self.priors_prescription = self.mtype

        # Instantiate a csalt model
        print('Making model')
        self.cm = model(self.mtype)
        if param is not None:
            self.cm.param = self.param

        # Define some fixed attributes for the modeling
        self.fixed_kw = {'restfreq': nu_rest, 'FOV': FOV, 'Npix': Npix, 'dist': dist, 'cfg_dict': cfg_dict} 

        # Import priors
        print('Initialising priors')
        self.priors = importlib.import_module('priors_'+self.priors_prescription)
        self.Ndim = len(self.priors.pri_pars)

        if image_plane:
            print('Initialising image plane comparison cube')
            from myfittingpackage import image_plane_fit as ipf 
            image = ipf(datacube=data_cube, distance=144.5, vismode=True, vel_range=vra_fit, npix=512, casa_sim=casa_sim)
            self.cm.image_plane_fit = image           


    def mcmc_fit(self):
        """
        Performs mcmc fit of data visibilities with models as specified in setup
        No extra parameters required
        """
        
        self.cm.sample_posteriors(self.msfile, kwargs=self.fixed_kw,
                         vra=self.vra_fit, restfreq=self.nu_rest, 
                         Nwalk=75, Nthreads=6, Ninits=10, Nsteps=50,
                         outpost='DMTau_EB3.DATA.h5', param=None)
        

    def initialise(self):
        """
        Initialises the mcmc such that you could pass in a theta array and obtain a
        single log likelihood value etc.
        """
        
        infdata = self.cm.fitdata(self.datafile, vra=self.vra_fit, vspacing=self.vspacing, restfreq=self.nu_rest)
        p0 = self.cm.mcfost_priors(self.priors, self.nwalk, self.Ndim)
        infdata = self.cm.cache(p0, infdata, self.nu_rest, self.fixed_kw)
        return infdata
    

    def get_probability(self, infdata, theta):
        """
        Returns the log probability of a specific theta array fitting the data
        Need to run initialise function first
        """

        global fdata
        global kw
        fdata = copy.deepcopy(infdata)
        kw = copy.deepcopy(self.fixed_kw)

                # Populate keywords from kwargs dictionary
        if 'restfreq' not in kw:
            kw['restfreq'] = self.nu_rest
        if 'FOV' not in kw:
            kw['FOV'] = 5.0
        if 'Npix' not in kw:
            kw['Npix'] = 256
        if 'dist' not in kw:
            kw['dist'] = 150.
        if 'chpad' not in kw:
            kw['chpad'] = 2
        if 'Nup' not in kw:
            kw['Nup'] = None
        if 'noise_inject' not in kw:
            kw['noise_inject'] = None
        if 'doppcorr' not in kw:
            kw['doppcorr'] = 'approx'
        if 'SRF' not in kw:
            kw['SRF'] = 'ALMA'


        loglikelihood, im_lnl = self.cm.log_likelihood(theta, fdata=fdata, kwargs=kw)

        priors = importlib.import_module('priors_'+self.priors_prescription)
        lnT = np.sum(priors.logprior(theta)) * fdata['Nobs']
        print('Priors', lnT)

        return loglikelihood+lnT
    

    def brute_force(self, data, datacube, single=False):

        """
        Needs to be called after initialise
        """

        global fdata
        global kw
        fdata = copy.deepcopy(data)
        kw = copy.deepcopy(self.fixed_kw)

        if len(self.param) > 2:
            print('Only maximum of two parameters to brute force over!')
            return
        

        # Populate keywords from kwargs dictionary
        if 'restfreq' not in kw:
            kw['restfreq'] = self.nu_rest
        if 'FOV' not in kw:
            kw['FOV'] = 5.0
        if 'Npix' not in kw:
            kw['Npix'] = 256
        if 'dist' not in kw:
            kw['dist'] = 150.
        if 'chpad' not in kw:
            kw['chpad'] = 2
        if 'Nup' not in kw:
            kw['Nup'] = None
        if 'noise_inject' not in kw:
            kw['noise_inject'] = None
        if 'doppcorr' not in kw:
            kw['doppcorr'] = 'approx'
        if 'SRF' not in kw:
            kw['SRF'] = 'ALMA'

        self.values = None
        self.datacube = datacube

        for param in self.param:
            values = []
            if param == 'inclination':
                for i in range(30):
                    values.append([6*i])
            elif param == 'stellar_mass':
                for i in range(50):
                    values.append([0.2 + 0.016*i])
            elif param == 'scale_height':
                for i in range(20):
                    values.append([10 + i])
            elif param == 'r_c':
                for i in range(20):
                    values.append([200 + 10*i])
            elif param == 'flaring_exp':
                for i in range(20):
                    values.append([1 + 0.05*i])
            elif param == 'PA':
                for i in range(20):
                    values.append([18*i])
            elif param == 'dust_param':
                for i in range(20):
                    values.append([10**(-5 + i/10)])
            elif param == 'vturb':
                for i in range(20):
                    values.append([0.01*i])
            elif param == 'gas_mass':
                for i in range(20):
                    values.append([10**(-2 + 0.05*i)])
            elif param == 'gasdust_ratio':
                for i in range(20):
                    values.append([10**(1+0.1*i)])
            else:
                print("Not a valid parameter")
                return

            if self.values is None:
                self.values = values
            else:
                full_values = []
                for value in self.values:
                    for second_value in values:
                        full_values.append([value[0], second_value[0]])
                self.values = full_values
        
        
        os.environ["OMP_NUM_THREADS"] = "1"

        if single:
            likelihoods = []
            for value in self.values:
                likelihoods.append(self.cm.log_likelihood(value, fdata, kw, None))
        
        else:
            with multiprocessing.Pool(processes=self.nthreads) as pool:
                likelihoods = pool.starmap(self.cm.log_likelihood, [(value, fdata, kw, None) for value in self.values])

        print(likelihoods)

        priors = importlib.import_module('priors_'+self.priors_prescription)
        ln_posteriors = []
        image_plane_likelihoods = []
        for i in range(len(likelihoods)):
            lnT = np.sum(priors.logprior(self.values[i])) * fdata['Nobs']
            ln_posteriors.append(likelihoods[i][0] + lnT)
            image_plane_likelihoods.append(likelihoods[i][1])

        name = ''
        for param in self.param:
            name += param
        stored_results = name + '_results.npz'
        np.savez(stored_results, self.values, ln_posteriors, image_plane_likelihoods)

        #self.movie_plot(self.values, ln_posteriors)

        # plt.figure()
        # plt.plot(self.values, ln_posteriors)
        # plt.title('Log posterior as a function of ' + self.param)
        # plt.xlabel(self.param)
        # plt.ylabel('Log posterior')
        # plt.savefig(self.param+'lnposterior.pdf')

        return ln_posteriors
    
    def movie_plot(self, vals, posteriors):

        for i in range(len(vals)):
            val = vals[i]
            fig = plt.figure(figsize=(30,6))
            subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1,2])
            axsLeft = subfigs[0].subplots(1, 1)
            axsLeft.plot(vals, posteriors)
            axsLeft.scatter([val], [posteriors[i]])
            axsLeft.title('Log posterior as a function of ' + self.param)
            axsLeft.xlabel(self.param)
            axsLeft.ylabel('Log posterior')
            

            if isinstance(self.datacube, str):
                print("Reading cube...")
                cube = casa.Cube(self.datacube, zoom=0.25)
                beam_area = cube.bmin * cube.bmaj * np.pi / (4.0 * np.log(2.0))
                pix_area = cube.pixelscale**2
            else:
                print('Need to give a cube to cube to compare with!')
                return 0
            
            #jobfs = os.getenv("JOBFS")
            directory = str(vals[i])+'_'+self.param
            model = mcfost.Line(directory+'/data_CO')
            residuals = mcfost.Line(directory+'/data_CO')

            velocities = model.velocity

            exocubelines = cube.image * pix_area/beam_area
            exocubelines[np.isnan(exocubelines)] = 0
            model_chans = []
            exocube_chans = []

            for vel in velocities:
                iv = np.abs(cube.velocity - vel).argmin()
                exocube_chans.append(exocubelines[iv])

            for vel in velocities:
                iv = np.abs(model.velocity - vel).argmin()
                model_chans.append(model.lines[iv])

            model_chans = np.array(model_chans)
            exocube_chans = np.array(exocube_chans)
            residuals.lines = exocube_chans - model_chans

            # Plotting arguments
            fmax = 0.05
            cmap = 'Blues'
            fmin = 0
            colorbar = False
            vlabel_color = 'black'
            lim = 6.99
            limits = [lim, -lim, -lim, lim]
            no_ylabel = False
            axsRight = subfigs[1].subplots(3, 9, sharex='all', sharey='all')
            axsRight.subplots_adjust(wspace=0.0, hspace=0.0)
            for i in range(9):
                if i != 0:
                    no_ylabel = True
                if i == 8:
                    colorbar = True
                if i != 4:
                    no_xlabel = True
                else:
                    no_xlabel = False
                #cube.plot(ax=axs[0, i], v=velocities[i], fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, vlabel_color=vlabel_color, limits=limits, no_xlabel=True, no_ylabel=True)
                #axs[0, i].get_xaxis().set_visible(False)
                #model.plot_map(ax=axs[1, i], v=velocities[i],  bmaj=cube.bmaj, bmin=cube.bmin, bpa=cube.bpa, fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, per_beam=True, limits=limits, no_xlabel=no_xlabel, no_ylabel=no_ylabel)

                cube.plot(ax=axsRight[0, i], v=velocities[i], fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, no_vlabel=False, vlabel_color='black', limits=limits, no_xlabel=True, no_ylabel=True)
                axsRight[0, i].get_xaxis().set_visible(False)
                print('Per beam')
                model.plot_map(ax=axsRight[1, i], v=velocities[i],  bmaj=cube.bmaj, bmin=cube.bmin, bpa=cube.bpa, fmin=fmin, fmax=fmax, cmap=cmap, colorbar=colorbar, per_beam=True, limits=limits, no_xlabel=True, no_ylabel=no_ylabel, no_vlabel=False, no_xticks=True)
                residuals.plot_map(ax=axsRight[2, i], v=velocities[i],  bmaj=cube.bmaj, bmin=cube.bmin, bpa=cube.bpa, fmin=-fmax, fmax=fmax, cmap='RdBu', colorbar=colorbar, per_beam=True, limits=limits, no_ylabel=True, no_vlabel=False, no_xlabel=no_xlabel)            
            
            
            plt.savefig(str(val)+'_'+self.param+'.pdf')

            return
        
    