import os
import sys
import time
import importlib
import numpy as np
import warnings
import copy
import casatools
from casatasks import (simobserve, concat)
from csalt.helpers import *
import scipy.constants as sc
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from scipy import linalg
from scipy import stats
from vis_sample import vis_sample
from vis_sample.classes import SkyImage
from multiprocess.pool import Pool
from math import isnan
import matplotlib.pyplot as plt


"""
    The dataset class for transporting visibility data.
"""
class dataset:

    def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # Spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK
        self.nchan = len(nu_TOPO)

        # Spatial frequencies in meters and lambda units
        self.um = um
        self.vm = vm
        self.ulam = self.um * np.mean(self.nu_TOPO) / sc.c
        self.vlam = self.vm * np.mean(self.nu_TOPO) / sc.c

        # Visibilities, weights, and timestamp IDs
        self.vis = vis
        self.wgt = wgt
        self.tstamp = tstamp_ID

        # Utility size trackers
        self.npol, self.nvis = vis.shape[0], vis.shape[2]
        self.nstamps = len(np.unique(tstamp_ID))

    def to_dict(self):

        return{
            'um': np.array(self.um),
            'vm': np.array(self.vm),
            'ulam': np.array(self.ulam),
            'vlam': np.array(self.vlam),
            'tstamp': np.array(self.tstamp),
            'nstamps': np.array(self.nstamps),
            'nu_TOPO': np.array(self.nu_TOPO),
            'nu_LSRK': np.array(self.nu_LSRK),
            'vis': np.array(self.vis),
            'wgt': np.array(self.wgt),
            'npol': np.array(self.npol),
            'nchan': np.array(self.nchan),
            'nvis': np.array(self.nvis),
        }


"""
    The model class that encapsulates the CSALT framework.
"""
class model:

    def __init__(self, prescription, path=None, quiet=True):

        if quiet:
            warnings.filterwarnings("ignore")

        if np.logical_or((path != os.getcwd()), (path is not None)):
            if path is not None:
                sys.path.append(path)
                self.path = path
        else:
            self.path = ''
        self.prescription = prescription
        self.param = None
        self.image_plane_fit = None


    """ 
        Generate a cube 
    """
    def cube(self, velax, pars, 
             restfreq=230.538e9, FOV=5.0, Npix=256, dist=150, cfg_dict={}):

        # Parse inputs
        if isinstance(velax, list): 
            velax = np.array(velax)
        
        fixed = restfreq, FOV, Npix, dist, cfg_dict

        # Load the appropriate prescription
        pfile = 'parametric_disk_'+self.prescription
        if not os.path.exists(pfile+'.py'):
            print('The prescription '+pfile+'.py does not exist.  Exiting.')
            sys.exit()
        pd = importlib.import_module(pfile)

        # Calculate the emission cube
        return pd.parametric_disk(velax, pars, fixed)


    """ 
        Spectral Response Functions (SRF) 
    """
    def SRF_kernel(self, srf_type, Nup=1):
        # Full-resolution cases for up-sampled spectra
        if Nup > 1:
            chix = np.arange(25 * Nup) / Nup
            xch = chix - np.mean(chix)
            
            # (F)XF correlators
            if srf_type in ['ALMA', 'VLA']:
                srf = 0.5 * np.sinc(xch) + \
                      0.25 * (np.sinc(xch - 1) + np.sinc(xch + 1))
            # FX correlators
            elif srf_type in ['SMA', 'NOEMA']:
                srf = (np.sinc(xch))**2
            # ALMA-WSU
            elif srf_type in ['ALMA-WSU']:
                _wsu = np.load('csalt/data/WSU_SRF.npz')
                wint = interp1d(_wsu['chix'], _wsu['srf'], 
                                fill_value='extrapolate', kind='cubic')
                srf = wint(xch)
            # break
            else:
                print('I do not know that SRF type.  Exiting.')
                sys.exit()

            return srf / np.trapz(srf, xch)

        # Approximations for sampled-in-place spectra
        else:
            # (F)XF correlators
            if srf_type in ['ALMA', 'VLA']:
                srf = np.array([0.00, 0.25, 0.50, 0.25, 0.00])
            # others
            else:
                srf = np.array([0.00, 0.00, 1.00, 0.00, 0.00])

            return srf
        

    """ Generate simulated data ('modeldict') """
    def modeldict(self, ddict, pars, kwargs=None):

        # Populate keywords from kwargs dictionary
        kw = {} if kwargs is None else kwargs
        if 'restfreq' not in kw:
            kw['restfreq'] = 230.538e9
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
        if 'cfg_dict' not in kw:
            kw['cfg_dict'] = {}
        if 'param' not in kw:
            kw['param'] = None
        if 'pb_attenuate' not in kw:
            kw['pb_attenuate'] = False

        # List of input EBs
        EBlist = range(ddict['Nobs'])

        # Copy the input data format to a model
        if kw['noise_inject'] is None:
            m_ = copy.deepcopy(ddict)
            for EB in EBlist:
                m_[str(EB)] = self.modelset(ddict[str(EB)], pars,
                                               restfreq=kw['restfreq'], 
                                               FOV=kw['FOV'], 
                                               Npix=kw['Npix'], 
                                               dist=kw['dist'], 
                                               chpad=kw['chpad'], 
                                               Nup=kw['Nup'],
                                               noise_inject=kw['noise_inject'],
                                               doppcorr=kw['doppcorr'], 
                                               SRF=kw['SRF'],
                                               cfg_dict=kw['cfg_dict'],
                                               param=kw['param'],
                                               attenuate=kwargs['pb_attenuate'])
            return m_
        else:
            p_, n_ = copy.deepcopy(ddict), copy.deepcopy(ddict)
            for EB in EBlist:
                p_[str(EB)], n_[str(EB)] = self.modelset(ddict[str(EB)], pars,
                                                restfreq=kw['restfreq'],
                                                FOV=kw['FOV'],
                                                Npix=kw['Npix'],
                                                dist=kw['dist'],
                                                chpad=kw['chpad'],
                                                Nup=kw['Nup'],
                                                noise_inject=kw['noise_inject'],
                                                doppcorr=kw['doppcorr'],
                                                SRF=kw['SRF'],
                                                cfg_dict=kw['cfg_dict'],
                                                param=kw['param'],
                                                attenuate=kwargs['pb_attenuate'])
            return p_, n_



    """ Generate simulated dataset ('modelset') """
    def modelset(self, dset, pars,
                 restfreq=230.538e9, FOV=5.0, Npix=256, dist=150, chpad=2, 
                 Nup=None, noise_inject=None, doppcorr='approx', SRF='ALMA',
                 gcf_holder=None, corr_cache=None, return_holders=False,
                 cfg_dict={}, icube=None, param=None, attenuate=False):

        """ Prepare the spectral grids: format = [timestamps, channels] """
        # Pad the LSRK frequencies
        dnu_n = (np.diff(dset.nu_LSRK, axis=1)[:,0])[:,None]
        _pad = (dset.nu_LSRK[:,0])[:,None] + \
                dnu_n * np.arange(-chpad, 0, 1)[None,:]
        pad_ = (dset.nu_LSRK[:,-1])[:,None] + \
                dnu_n * np.arange(1, chpad+1, 1)[None,:]
        nu_ = np.concatenate((_pad, dset.nu_LSRK, pad_), axis=1)

        # Upsample the LSRK frequencies (if requested)
        if Nup is not None:
            nchan = nu_.shape[1]
            nu = np.empty((dset.nstamps, (nchan - 1) * Nup + 1))
            for it in range(dset.nstamps):
                nu[it,:] = np.interp(np.arange((nchan - 1) * Nup + 1),
                                     np.arange(0, nchan * Nup, Nup), nu_[it,:])
        else:
            nu, Nup = 1. * nu_, 1
        nch = nu.shape[1]

        # Calculate LSRK velocities
        vel = sc.c * (1 - nu / restfreq)

        ### - Compute the model visibilities
        mvis_ = np.squeeze(np.empty((dset.npol, nch, dset.nvis, 2)))

        exclude = ['MCFOST', 'FITS', 'MYFITS']

        if self.prescription in exclude:
            if param is not None and param[-1] == 'mu_DEC':
                mu_RA = pars[-2]
                mu_DEC = pars[-1]               
            else:
                mu_RA = 0
                mu_DEC = 0
        else:
            mu_RA = pars[-2]
            mu_DEC = pars[-1]

        if param is not None:
            #param_dict_form = {param: pars[0]}
            param_dict_form = {}
            for i in range(len(param)):
                param_dict_form[param[i]] = pars[i]
            pars = param_dict_form

        image_lnl = None

        # *Exact* Doppler correction calculation
        if doppcorr == 'exact':
            print('exact')
            for itime in range(dset.nstamps):
                # track the steps
                print('timestamp '+str(itime+1)+' / '+str(dset.nstamps))

                # make a cube
                if icube is None:
                    icube = self.cube(vel[itime,:], pars, restfreq=restfreq,
                                      FOV=FOV, Npix=Npix, dist=dist, 
                                      cfg_dict=cfg_dict)
                    if isinstance(icube, dict):
                        image_lnl = icube['image_lnl']
                        icube = icube['cube']

                # visibility indices for this timestamp only
                ixl = np.min(np.where(dset.tstamp == itime))
                ixh = np.max(np.where(dset.tstamp == itime)) + 1

                # sample the FFT on the (u, v) spacings
                mvis = vis_sample(imagefile=copy.deepcopy(icube), 
                                  uu=dset.ulam[ixl:ixh], vv=dset.vlam[ixl:ixh],
                                  gcf_holder=gcf_holder, corr_cache=corr_cache,
                                  mu_RA=mu_RA, mu_DEC=mu_DEC, 
                                  mod_interp=False).T

                # populate the results in the output array *for this stamp*
                mvis_[0,:,ixl:ixh,0] = mvis.real
                mvis_[1,:,ixl:ixh,0] = mvis.real
                mvis_[0,:,ixl:ixh,1] = mvis.imag
                mvis_[1,:,ixl:ixh,1] = mvis.imag

        elif doppcorr == 'approx':
            # velocities at the mid-point timestamp of this EB
            v_model = vel[int(np.round(nu.shape[0] / 2)),:]

            # make a cube
            if icube is None:
                icube = self.cube(v_model, pars, restfreq=restfreq,
                                  FOV=FOV, Npix=Npix, dist=dist, cfg_dict=cfg_dict)
                if isinstance(icube, dict):
                    image_lnl = icube['image_lnl']
                    icube = icube['cube']
            
            if attenuate:
                print('attenuating')
                diameter = dset.diameter #? need to implement
                pixel_scale = np.abs(np.diff(icube.ra)[0]) # something
                print(pixel_scale)
                to_vis_sample = copy.deepcopy(icube)
                to_vis_sample.data = apply_primary_beam_effect(icube.data, pixel_scale, restfreq, diameter)
            else:
                to_vis_sample = icube

            # sample the FFT on the (u, v) spacings
            if return_holders:
                mvis, gcf, corr = vis_sample(imagefile=copy.deepcopy(to_vis_sample), 
                                             uu=dset.ulam, vv=dset.vlam,
                                             mu_RA=mu_RA, mu_DEC=mu_DEC,
                                             return_gcf=True, 
                                             return_corr_cache=True,
                                             mod_interp=False)
                
                return mvis.T, gcf, corr, icube, image_lnl
            else:
                mvis = vis_sample(imagefile=copy.deepcopy(to_vis_sample), uu=dset.ulam, vv=dset.vlam,
                                  gcf_holder=gcf_holder, corr_cache=corr_cache,
                                  mu_RA=mu_RA, mu_DEC=mu_DEC, 
                                  mod_interp=False).T

            # distribute to different timestamps by interpolation
            for itime in range(dset.nstamps):
                ixl = np.min(np.where(dset.tstamp == itime))
                ixh = np.max(np.where(dset.tstamp == itime)) + 1
                fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind='cubic',
                                fill_value='extrapolate')
                interp_vis = fint(vel[itime,:])
                mvis_[0,:,ixl:ixh,0] = interp_vis.real
                mvis_[1,:,ixl:ixh,0] = interp_vis.real
                mvis_[0,:,ixl:ixh,1] = interp_vis.imag
                mvis_[1,:,ixl:ixh,1] = interp_vis.imag
        elif doppcorr is None:
            print('I AM NOT DOING A DOPPLER CORRECTION!')
            # make a cube
            if icube is None:
                icube = self.cube(vel[0,:], pars, restfreq=restfreq,
                              FOV=FOV, Npix=Npix, 
                              dist=dist, cfg_dict=cfg_dict)
                if isinstance(icube, dict):
                    image_lnl = icube['image_lnl']
                    icube = icube['cube']

            # sample the FFT on the (u, v) spacings
            if return_holders:
                mvis, gcf, corr = vis_sample(imagefile=copy.deepcopy(icube), 
                                             uu=dset.ulam, vv=dset.vlam,
                                             mu_RA=mu_RA, mu_DEC=mu_DEC,
                                             return_gcf=True,
                                             return_corr_cache=True,
                                             mod_interp=False)
                return mvis.T, gcf, corr, icube, image_lnl
            else:
                mvis = vis_sample(imagefile=copy.deepcopy(icube), uu=dset.ulam, vv=dset.vlam,
                                  gcf_holder=gcf_holder, corr_cache=corr_cache,
                                  mu_RA=mu_RA, mu_DEC=mu_DEC,
                                  mod_interp=False).T
        else:
            print('You need to specify a doppcorr method.  Exiting.')
            sys.exit()

        # Convolve with the spectral response function (SRF)
        if SRF is not None:
            kernel = self.SRF_kernel(SRF, Nup=Nup)
            mvis_pure = convolve1d(mvis_, kernel, axis=1, mode='nearest')
        else:
            print('I AM NOT DOING AN SRF CONVOLUTION!')
            mvis_pure = 1. * mvis_

        # Decimate and package the pure visibility spectra
        mvis_pure = mvis_pure[:,::Nup,:,:]
        mvis_pure = mvis_pure[:,chpad:-chpad,:,:]
        mvis_p = mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
        mset_p = dataset(dset.um, dset.vm, mvis_p, dset.wgt, dset.nu_TOPO,
                         dset.nu_LSRK, dset.tstamp)

        # Return the pure or pure and noisy models
        if noise_inject is None:
            return mset_p, icube, image_lnl
        else:
            # Calculate noise spectra
            noise = self.calc_noise(noise_inject, dset, 
                                    nchan=nch, Nup=Nup, SRF=SRF)

            # SRF convolution of noisy data
            if SRF is not None:
                mvis_noisy = convolve1d(mvis_ + noise, kernel, axis=1, 
                                        mode='nearest')
            else:
                mvis_noisy = mvis_ + noise

            # Decimate and package the pure visibility spectra
            mvis_noisy = mvis_noisy[:,::Nup,:,:]
            mvis_noisy = mvis_noisy[:,chpad:-chpad,:,:]
            mvis_n = mvis_noisy[:,:,:,0] + 1j * mvis_noisy[:,:,:,1]
            mset_n = dataset(dset.um, dset.vm, mvis_n, dset.wgt, dset.nu_TOPO,
                             dset.nu_LSRK, dset.tstamp)
            return (mset_p, icube, image_lnl), (mset_n, icube, image_lnl)


    """
        A noise calculator
    """
    def calc_noise(self, noise_inject, dataset, nchan=1, Nup=None, SRF='ALMA'):

        # Scale input RMS for desired noise per vis-chan-pol
        sigma_out = noise_inject * np.sqrt(dataset.npol * dataset.nvis)

        # Scale to account for spectral up-sampling and SRF (TEMPORARY)
        if Nup is None: Nup = 1
        if SRF in ['ALMA', 'VLA']:
            fcov = 8./3.
        else:
            fcov = 1.
        sigma_noise = sigma_out * np.sqrt(Nup * fcov)

        # Random Gaussian noise draws
        noise = np.random.normal(0, sigma_noise,
                                 (dataset.npol, nchan, dataset.nvis, 2))
        
        return np.squeeze(noise)



    """
        Create a blank MS template 
    """
    def template_MS(self, msfile, config='', t_total='1min', 
                    sim_save=False, RA='16:00:00.00', DEC='-30:00:00.00', 
                    restfreq=230.538e9, dnu_native=122e3, V_span=10e3, 
                    V_tune=0.0e3, t_integ='6s', HA_0='0h', date='2023/03/20',
                    observatory='ALMA', force_to_LSRK=False):

        # Load the measures tools
        me = casatools.measures()

        # Parse / determine the executions
        if np.isscalar(config): config = np.array([config])
        Nobs = len(config)

        # things to format check:
                # RA, DEC have full hms/dms string formatting
                # HA_0 has proper string formatting
                # date has proper string formatting
                # msfile has proper '.ms' ending

        # If only scalars specified for keywords, copy them for each execution
        if np.isscalar(t_total): t_total = np.repeat(t_total, Nobs)
        if np.isscalar(dnu_native): dnu_native = np.repeat(dnu_native, Nobs)
        if np.isscalar(V_span): V_span = np.repeat(V_span, Nobs)
        if np.isscalar(V_tune): V_tune = np.repeat(V_tune, Nobs)
        if np.isscalar(HA_0): HA_0 = np.repeat(HA_0, Nobs)
        if np.isscalar(date): date = np.repeat(date, Nobs)
        if np.isscalar(t_integ): t_integ = np.repeat(t_integ, Nobs)

        # Move to simulation space
        cwd = os.getcwd()
        out_path, out_name = os.path.split(msfile)
        if out_path != '':
            if not os.path.exists(out_path): os.system('mkdir '+out_path)
            os.chdir(out_path)

        # Loop over execution blocks
        obs_files = []
        for i in range(Nobs):

            # Calculate the number of native channels
            nch = 2 * int(V_span[i] / (sc.c * dnu_native[i] / restfreq)) + 1

            # Calculate the LST starting time of the execution block
            h, m, s = RA.split(':')
            LST_h = int(h) + int(m)/60 + float(s)/3600 + float(HA_0[i][:-1])
            LST_0 = str(datetime.timedelta(hours=LST_h))
            if (LST_h < 10.): LST_0 = '0' + LST_0

            # Get the observatory longitude
            obs_long = np.degrees(me.observatory(observatory)['m0']['value'])

            # Calculate the UT starting time of the execution block
            UT_0 = LST_to_UTC(date[i], LST_0, obs_long)

            # Calculate the TOPO tuning frequency
            nu_tune_0 = doppler_set(restfreq, V_tune[i], UT_0, RA, DEC,
                                    observatory=observatory)

            # Generate a dummy (empty) cube
            ia = casatools.image()
            dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])
            ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)
            ia.done()

            # Compute the midpoint HA
            if (t_total[i][-1] == 'h'):
                tdur = float(t_total[i][:-1])
            elif (t_total[i][-3:] == 'min'):
                tdur = float(t_total[i][:-3]) / 60
            elif (t_total[i][-1] == 's'):
                tdur = float(t_total[i][:-1]) / 3600
            HA_mid = str(float(HA_0[i][:-1]) + 0.5 * tdur) +'h'

            # Generate the template sub-MS file
            simobserve(project=out_name[:-3]+'_'+str(i)+'.sim',
                       skymodel='dummy.image',
                       antennalist=config[i],
                       totaltime=t_total[i],
                       integration=t_integ[i],
                       thermalnoise='',
                       hourangle=HA_mid,
                       indirection='J2000 '+RA+' '+DEC,
                       refdate=date[i],
                       incell='0.01arcsec',
                       mapsize='5arcsec',
                       incenter=str(nu_tune_0 / 1e9)+'GHz',
                       inwidth=str(dnu_native[i] * 1e-3)+'kHz',
                       outframe='TOPO')

            # Pull the sub-MS file out of the simulation directory
            cfg_dir, cfg_file = os.path.split(config[i])
            sim_MS = out_name[:-3]+'_'+str(i)+'.sim/'+out_name[:-3]+ \
                     '_'+str(i)+'.sim.'+cfg_file[:-4]+'.ms'
            os.system('rm -rf '+out_name[:-3]+'_'+str(i)+'.ms*')
            os.system('mv '+sim_MS+' '+out_name[:-3]+'_'+str(i)+'.ms')

            # Delete the simulation directory if requested
            if not sim_save:
                os.system('rm -rf '+out_name[:-3]+'_'+str(i)+'.sim')

            # Delete the dummy (empty) cube
            os.system('rm -rf dummy.image')

            # Update the file list
            obs_files += [out_name[:-3]+'_'+str(i)+'.ms']

        # Concatenate the sub-MS files into a single MS
        os.system('rm -rf '+out_name[:-3]+'.ms*')
        if Nobs > 1:
            concat(vis=obs_files,
                   concatvis=out_name[:-3]+'.ms',
                   dirtol='0.1arcsec',
                   copypointing=False)
        else:
            os.system('mv '+out_name[:-3]+'_0.ms '+out_name[:-3]+'.ms')

        # Clean up
        os.system('rm -rf '+out_name[:-3]+'_*.ms*')
        os.chdir(cwd)

        return None


    """
        Function to parse and package visibility data for inference
    """
    def fitdata(self, msfile, vra=None, vcensor=None, 
                vspacing=None, restfreq=230.538e9, chbin=1, 
                well_cond=300, eb=None):

        # Load the data from the MS file into a dictionary

        # check that the data hasn't already been read in - this saves a lot of memory
        ebs = 0
        current_dir = os.getcwd()
        for filename in os.listdir(current_dir):
            if filename.startswith('eb') and filename.endswith('npz'):
                ebs +=1
        
        if ebs != 0:
            infdata = self.read_in_data(ebs, chbin, eb)
            return infdata

        data_dict = read_MS(msfile)

        # If chbin is a scalar, distribute it over the Nobs executions
        if np.isscalar(chbin):
            chbin = chbin * np.ones(data_dict['Nobs'], dtype=int)
        else:
            if isinstance(chbin, list):
                chbin = np.asarray(chbin)

        # If vra is a list, make it an array
        if isinstance(vra, list):
            vra = np.asarray(vra)

        # Assign an output dictionary
        out_dict = {'Nobs': data_dict['Nobs'], 'chbin': chbin}

        # Force chbin <= 2
        if np.any(chbin > 2):
            print('Forcing chbin --> 2; do not over-bin your data!')
        chbin[chbin > 2] = 2

        if self.prescription == 'MCFOST':
            min_nchan = self.find_min_chan(data_dict, vra, restfreq, chbin) 

        # Loop over executions
        for i in range(data_dict['Nobs']):

            presaved_dict = 'eb' + str(i) + '.npz'
            if os.path.isfile(presaved_dict):
                loaded_array = np.load(presaved_dict)
                um = loaded_array['arr_0']
                vm = loaded_array['arr_1']
                vis = loaded_array['arr_2']
                wgt = loaded_array['arr_3']
                nu_TOPO = loaded_array['arr_4']
                nu_LSRK = loaded_array['arr_5']
                tstamp = loaded_array['arr_6']
                invcov = loaded_array['arr_7']
                lnL0 = loaded_array['arr_8']
            
                out_dict[str(i)] = dataset(um, vm, vis, wgt,
                                           nu_TOPO, nu_LSRK, tstamp)
                
                # Package additional information into the dictionary
                out_dict['invcov_'+str(i)] = invcov
                out_dict['lnL0_'+str(i)] = lnL0
                out_dict['gcf_'+str(i)] = None
                out_dict['corr_'+str(i)] = None
                
            else:

                global scov
                global _wgt

                # Pull the dataset object for this execution
                data = data_dict[str(i)]

                # If necessary, distribute weights across the spectrum
                if not data.wgt.shape == data.vis.shape:
                    data.wgt = np.tile(data.wgt, (data.nchan, 1, 1))
                    data.wgt = np.rollaxis(data.wgt, 1, 0)

                # Convert the LSRK frequency grid to velocities
                v_LSRK = sc.c * (1 - data.nu_LSRK / restfreq)

                # Fix direction of desired velocity bounds
                if vra is None: vra = np.array([-1e5, 1e5])
                dv, dvra = np.diff(v_LSRK, axis=1), np.diff(vra)
                if np.logical_or(np.logical_and(np.all(dv < 0), np.all(dvra > 0)),
                                 np.logical_and(np.all(dv < 0), np.all(dvra < 0))):
                    vra_ = vra[::-1]
                else:
                    vra_ = 1. * vra
                sgn_v = np.sign(np.diff(vra_)[0])

                # Find where to clip to lie within the desired velocity bounds
                midstamp = int(data.nstamps / 2)
                ixl = np.abs(v_LSRK[midstamp,:] - vra_[0]).argmin()
                ixh = np.abs(v_LSRK[midstamp,:] - vra_[1]).argmin()

                # Adjust indices to ensure they are evenly divisible by chbin
                if self.prescription != 'MCFOST':
                    if np.logical_and((chbin[i] > 1), ((ixh - ixl) % chbin[i] != 0)):
                        # bounded at upper edge only
                        if np.logical_and((ixh == (data.nchan - 1)), (ixl > 0)):
                            ixl -= 1
                        # bounded at lower edge only
                        elif np.logical_and((ixh < (data.nchan - 1)), (ixl == 0)):
                            ixh += 1
                        # bounded at both edges
                        elif np.logical_and((ixh == (data.nchan - 1)), (ixl == 0)):
                            ixh -= 1
                        # unbounded on either side
                        else:
                            ixh += 1
                else:
                    # for the mcfost models all the ebs need to have the same number of channels
                    if (ixh-ixl) != min_nchan:
                        diff_lower = np.abs(v_LSRK[midstamp, ixl] - vra[0])
                        diff_higher = np.abs(v_LSRK[midstamp, ixh] - vra[1])
                        if np.logical_and((diff_lower > diff_higher), (ixl > 0)):
                            ixl = ixh - min_nchan
                        else:
                            ixh = ixl + min_nchan


                # Clip the data to cover only the frequencies of interest
                inu_TOPO = data.nu_TOPO[ixl:ixh]
                inu_LSRK = data.nu_LSRK[:,ixl:ixh]
                iv_LSRK = v_LSRK[:,ixl:ixh] 
                inchan = inu_LSRK.shape[1]
                ivis = data.vis[:,ixl:ixh,:]
                iwgt = data.wgt[:,ixl:ixh,:]


                # Binning operations
                print('Binning Time')
                binned = True if chbin[i] > 1 else False
                if binned:
                    bnchan = int(inchan / chbin[i])
                    bshape = (data.npol, -1, chbin[i], data.nvis)
                    print(iwgt.shape)
                    wt = iwgt.reshape(bshape)
                    bvis = np.average(ivis.reshape(bshape), weights=wt, axis=2)
                    bwgt = np.sum(wt, axis=2)

                # Channel censoring
                if vcensor is not None:
                    cens_chans = np.ones(inchan, dtype='bool')
                    if not any(isinstance(el, list) for el in vcensor):
                        vcensor = [vcensor]
                    for j in range(len(vcensor)):
                        if sgn_v < 0:
                            vcens = (vcensor[j])[::-1]
                        else:
                            vcens = vcensor[j]
                        cixl = np.abs(iv_LSRK[midstamp,:] - vcens[0]).argmin()
                        cixh = np.abs(iv_LSRK[midstamp,:] - vcens[1]).argmin()
                        cens_chans[cixl:cixh+1] = False
                    iwgt[:,cens_chans == False,:] = 0

                    if binned:
                        bcens_chans = np.all(cens_chans.reshape((-1, chbin[i])),
                                         axis=1)
                        bwgt[:,bcens_chans == False,:] = 0


                 # Select specific channels
                if vspacing is not None:
                    if np.abs(vra[1]-vra[0]) % vspacing != 0:
                        print('Channel range not divisible by spacing factor!')
                        return
                    else:
                        nchans = int((np.abs(vra[1]-np.abs(vra[0])))/(vspacing) + 1)
                        vkeep = []
                        for chan_i in range(nchans):
                            vkeep.append(int(((np.abs(iv_LSRK[midstamp, :]-(vra[0]+(chan_i*vspacing))).argmin()))))

                        print('keeping ', len(vkeep))

                        inu_TOPO = inu_TOPO[vkeep]
                        inu_LSRK = inu_LSRK[:, vkeep]
                        iv_LSRK = iv_LSRK[:, vkeep]
                        inchan = inu_LSRK.shape[1]
                        ivis = ivis[:, vkeep, :]
                        iwgt = iwgt[:, vkeep, :]

                # Pre-calculate the spectral covariance matrix 
                # (** note: this assumes the Hanning kernel for ALMA **)
                if binned:
                    scov = (5/16) * np.eye(bnchan) \
                           + (3/32) * (np.eye(bnchan, k=-1) + np.eye(bnchan, k=1))
                else:
                    scov = (3/8) * np.eye(inchan) \
                           + (1/4) * (np.eye(inchan, k=-1) + np.eye(inchan, k=1)) \
                           + (1/16) * (np.eye(inchan, k=-2) + np.eye(inchan, k=2))

                # If well-conditioned (usually for binned), do direct inversion
                if np.linalg.cond(scov) <= well_cond:
                    print('EB '+str(i)+' SCOV inverted with direct calculation.')
                    scov_inv = linalg.inv(scov)

                # See if you can use Cholesky factorization
                else:
                    chol = linalg.cholesky(scov)
                    if np.linalg.cond(chol) <= well_cond:
                        print('EB '+str(i)+' SCOV inverted with Cholesky'
                              + ' factorization')
                        scov_inv = np.dot(linalg.inv(chol), linalg.inv(chol.T))

                    # Otherwise use SVD
                    else:
                        print('EB '+str(i)+' SCOV inverted with singular value'
                              + ' decomposition')
                        uu, ss, vv = linalg.svd(scov)
                        scov_inv = np.dot(vv.T, np.dot(np.diag(ss**-1), uu.T))

                # Pre-calculate the log-likelihood normalization term
                print('Log likelihood normalisation term time!')
                dterm = np.empty((data.npol, data.nvis))
                print('total: ', data.nvis)
                _wgt = bwgt if binned else iwgt
                # for ii in range(data.nvis):
                #     print(ii, '/', range(data.nvis))
                #     for jj in range(data.npol):
                #         _wgt = bwgt[jj,:,ii] if binned else iwgt[jj,:,ii]
                #         sgn, lndet = np.linalg.slogdet(scov / _wgt)
                #         dterm[jj,ii] = sgn * lndet
                input_args = [(ii, jj) for ii in range(data.nvis) for jj in range(data.npol)]
                with Pool() as p:
                    print("Starting multiprocessing")
                    results = p.map(determinant, input_args)
                for result in results:
                    jj = result[0]
                    ii = result[1]
                    det = result[2]
                    dterm[jj, ii] = det                
                _ = np.prod(bvis.shape) if binned else np.prod(ivis.shape)
                lnL0 = -0.5 * (_ * np.log(2 * np.pi) + np.sum(dterm))

                # Package the output data into the dictionary
                if binned:
                    odata = dataset(data.um, data.vm, bvis, bwgt, inu_TOPO,
                                    inu_LSRK, data.tstamp)
                    np.savez(presaved_dict, data.um, data.vm, bvis, bwgt,
                             inu_TOPO, inu_LSRK, data.tstamp, scov_inv, lnL0)
                else:
                    odata = dataset(data.um, data.vm, ivis, iwgt, inu_TOPO,
                                    inu_LSRK, data.tstamp)
                    np.savez(presaved_dict, data.um, data.vm, ivis, iwgt,
                             inu_TOPO, inu_LSRK, data.tstamp, scov_inv, lnL0) 
                out_dict[str(i)] = odata

                # Package additional information into the dictionary
                out_dict['invcov_'+str(i)] = scov_inv
                out_dict['lnL0_'+str(i)] = lnL0
                out_dict['gcf_'+str(i)] = None
                out_dict['corr_'+str(i)] = None

        # Return the output dictionary
        return out_dict
    
    """
        Read in data that has already been formatted correctly

    """

    def read_in_data(self, ebs, chbin, eb=None):

        # If chbin is a scalar, distribute it over the Nobs executions
        if np.isscalar(chbin):
            chbin = chbin * np.ones(ebs, dtype=int)
        else:
            if isinstance(chbin, list):
                chbin = np.asarray(chbin)

        # Assign an output dictionary
        out_dict = {}
        out_dict['Nobs'] = ebs
        out_dict['chbin'] = chbin

        if eb is not None:
            # eb needs to be a list
            ebs = eb
        else:
            ebs = [i for i in range(ebs)]

        # Loop over executions
        for i in range(len(ebs)):
            
            presaved_dict = 'eb' + str(ebs[i]) + '.npz'
            loaded_array = np.load(presaved_dict)
            um = loaded_array['arr_0']
            vm = loaded_array['arr_1']
            vis = loaded_array['arr_2']
            wgt = loaded_array['arr_3']
            nu_TOPO = loaded_array['arr_4']
            nu_LSRK = loaded_array['arr_5']
            tstamp = loaded_array['arr_6']
            invcov = loaded_array['arr_7']
            lnL0 = loaded_array['arr_8']
            
            out_dict[str(i)] = dataset(um, vm, vis, wgt,
                                        nu_TOPO, nu_LSRK, tstamp)
            
            # Package additional information into the dictionary
            out_dict['invcov_'+str(i)] = invcov
            out_dict['lnL0_'+str(i)] = lnL0
            out_dict['gcf_'+str(i)] = None
            out_dict['corr_'+str(i)] = None

        return out_dict
        

    
    """
        Find the minimum number of channels needed to model the data (for Caitlyn's MCFOST models)

    """
    def find_min_chan(self, data_dict, vra, restfreq, chbin):
        
        min_nchan = None

         # Loop over executions
        for i in range(data_dict['Nobs']):

            # Pull the dataset object for this execution
            data = data_dict[str(i)]

            # If necessary, distribute weights across the spectrum
            if not data.wgt.shape == data.vis.shape:
                data.wgt = np.tile(data.wgt, (data.nchan, 1, 1))
                data.wgt = np.rollaxis(data.wgt, 1, 0)

            # Convert the LSRK frequency grid to velocities
            v_LSRK = sc.c * (1 - data.nu_LSRK / restfreq)

            # Fix direction of desired velocity bounds
            if vra is None: vra = np.array([-1e5, 1e5])
            dv, dvra = np.diff(v_LSRK, axis=1), np.diff(vra)
            if np.logical_or(np.logical_and(np.all(dv < 0), np.all(dvra > 0)),
                             np.logical_and(np.all(dv < 0), np.all(dvra < 0))):
                vra_ = vra[::-1]
            else:
                vra_ = 1. * vra
            sgn_v = np.sign(np.diff(vra_)[0])

            # Find where to clip to lie within the desired velocity bounds
            midstamp = int(data.nstamps / 2)
            ixl = np.abs(v_LSRK[midstamp,:] - vra_[0]).argmin()
            ixh = np.abs(v_LSRK[midstamp,:] - vra_[1]).argmin()

            # Adjust indices to ensure they are evenly divisible by chbin
            if np.logical_and((chbin[i] > 1), ((ixh - ixl) % chbin[i] != 0)):
                # bounded at upper edge only
                if np.logical_and((ixh == (data.nchan - 1)), (ixl > 0)):
                    ixl -= 1
                # bounded at lower edge only
                elif np.logical_and((ixh < (data.nchan - 1)), (ixl == 0)):
                    ixh += 1
                # bounded at both edges
                elif np.logical_and((ixh == (data.nchan - 1)), (ixl == 0)):
                    ixh -= 1
                # unbounded on either side
                else:
                    ixh += 1

            if min_nchan is None or (ixh -ixl) < min_nchan:
                min_nchan = ixh - ixl

        #if min_nchan % 2 == 0:
        #    min_nchan +=1

        while min_nchan % chbin[i] != 0:
            min_nchan += 1

        print(min_nchan)
        
        return min_nchan


    """
        Sample the posteriors.
    """
    def sample_posteriors(self, msfile, vra=None, vcensor=None, vspacing=None, kwargs=None,
                          restfreq=230.538e9, chbin=1, well_cond=300,
                          Nwalk=75, Ninits=20, Nsteps=1000, 
                          outpost='stdout.h5', append=False, Nthreads=6, param=None, 
                          eb_selected=None, initial=None, reinitialize=False):
        
        import emcee
        from multiprocessing import Pool


        # if Nthreads > 1:
        #     os.environ["OMP_NUM_THREADS"] = "1"

        # Parse the data into proper format
        print('Fitting data')
        infdata = self.fitdata(msfile, vra=vra, vcensor=vcensor, 
                                vspacing=vspacing, restfreq=restfreq, chbin=chbin, 
                                well_cond=well_cond, eb=eb_selected)
        
        if param is not None:
            self.param = param
            self.priors_prescription = self.prescription
            # for parameter in self.param:
            #     self.priors_prescription += '_'+parameter
        else:
            self.priors_prescription = self.prescription

        if 'pb_attenuate' not in kwargs:
            kwargs['pb_attenuate'] = False
        else:
            if 'diameters' not in kwargs:
                print('We need to include diameters to implement primary beam attenuation!')
                kwargs['pb_attenuate'] = False
            else:
                for EB in range(infdata['Nobs']):
                    infdata[str(EB)].diameter = kwargs['diameters'][EB]

        # Initialize the parameters using random draws from the priors
        print('Initialising priors')
        priors = importlib.import_module('priors_'+self.priors_prescription)
        Ndim = len(priors.pri_pars)
        if self.prescription == 'MCFOST':
            p0 = self.mcfost_priors(priors, Nwalk, Ndim, initial)
        else:
            p0 = np.empty((Nwalk, Ndim))
            for ix in range(Ndim):
                if ix == 9:
                    p0[:,ix] = np.sqrt(2 * sc.k * p0[:,6] / (28 * (sc.m_p+sc.m_e)))
                else:
                    _ = [str(priors.pri_pars[ix][ip])+', '
                         for ip in range(len(priors.pri_pars[ix]))]
                    cmd = 'np.random.'+priors.pri_types[ix]+ \
                          '('+"".join(_)+str(Nwalk)+')'
                    p0[:,ix] = eval(cmd)

        # Acquire and store the GCF and CORR caches for iterative sampling
        print('Caching...')
        infdata = self.cache(p0, infdata, restfreq, kwargs)

        # Declare the data and kwargs as globals (for speed in pickling)
        global fdata
        global kw
        fdata = copy.deepcopy(infdata)
        kw = copy.deepcopy(kwargs)

        # Populate keywords from kwargs dictionary
        if 'restfreq' not in kw:
            kw['restfreq'] = restfreq
        if 'FOV' not in kw:
            kw['FOV'] = 5.0
        if 'Npix' not in kw:
            kw['Npix'] = 256
        if 'dist' not in kw:
            kw['dist'] = 150.
        if 'cfg_dict' not in kw:
            kw['cfg_dict'] = {}
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

 

        if not append:
            if initial is not None:
                 # Sample the posterior distribution
                print('Full MCMC')
                backend = emcee.backends.HDFBackend(outpost)
                backend.reset(Nwalk, Ndim)
                with Pool(processes=Nthreads) as pool:
                    samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool, backend=backend)
                    t0 = time.time()
                    samp.run_mcmc(p0, Nsteps, progress=True)
                t1 = time.time()
                print('backend run in ', t1-t0)
            else:
                # Initialize the MCMC walkers
                print('Initialising walkers')
                with Pool(processes=Nthreads) as pool:
                    isamp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool)
                    isamp.run_mcmc(p0, Ninits, progress=True)
                isamples = isamp.get_chain()   # [Ninits, Nwalk, Ndim]-shaped
                lop0 = np.quantile(isamples[-1, :, :], 0.25, axis=0)
                hip0 = np.quantile(isamples[-1, :, :], 0.75, axis=0)
                # normal centred on median value with a sigma that depends on each parameter
                p00 = [np.random.uniform(lop0, hip0, Ndim) for iw in range(Nwalk)]

                # Prepare the backend
                os.system('rm -rf '+outpost)
                backend = emcee.backends.HDFBackend(outpost)
                backend.reset(Nwalk, Ndim)

                # Sample the posterior distribution
                print('Full MCMC')
                with Pool(processes=Nthreads) as pool:
                    samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool, backend=backend)
                    t0 = time.time()
                    samp.run_mcmc(p00, Nsteps, progress=True)
                t1 = time.time()
                print('backend run in ', t1-t0)
        else:
            if reinitialize:
                print('Reinitializing...')
                # Load the old backend
                new_backend = emcee.backends.HDFBackend(outpost)
                print("Initial size: {0}".format(new_backend.iteration))
                isamp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior, backend=new_backend)
                samples = isamp.get_chain()   # [Ninits, Nwalk, Ndim]-shaped
                lop0 = np.quantile(samples[-1, :, :], 0.25, axis=0)
                hip0 = np.quantile(samples[-1, :, :], 0.75, axis=0)
                # normal centred on median value with a sigma that depends on each parameter
                p0 = [np.random.uniform(lop0, hip0, Ndim) for iw in range(Nwalk)]
                print('Restarting run...')
                with Pool(processes=Nthreads) as pool:
                    samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool, backend=new_backend)
                    t0 = time.time()
                    samp.run_mcmc(p0, Nsteps-new_backend.iteration, progress=True)
                t1 = time.time()
            else:
                # Load the old backend
                new_backend = emcee.backends.HDFBackend(outpost)
                print("Initial size: {0}".format(new_backend.iteration))
                
                # Continue sampling the posterior distribution
                with Pool(processes=Nthreads) as pool:
                    samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                                pool=pool, backend=new_backend)
                    t0 = time.time()
                    samp.run_mcmc(None, Nsteps-new_backend.iteration, progress=True)
                t1 = time.time()

        print('\n\n    This run took %.2f hours' % ((t1 - t0) / 3600))

        # Release the globals
        del fdata
        del kw

        return samp
    
    """
    Sample the posteriors with nested sampler ultranest
    """
    
    def sample_posteriors_ultranest(self, msfile, vra=None, vcensor=None, 
                                    vspacing=None, kwargs=None, restfreq=230.538e9, 
                                    chbin=1, well_cond=300, outpost='stdout.h5', 
                                    resume='subfolder', run_num=None, param=None, trace=False,
                                    eb_selected=None):

        import ultranest

        # Parse the data into proper format
        print('Fitting data')
        infdata = self.fitdata(msfile, vra=vra, vcensor=vcensor, 
                               vspacing=vspacing, restfreq=restfreq, chbin=chbin, 
                               well_cond=well_cond, eb=eb_selected)
    
        if param is None:
            param = ['inclination', 'stellar_mass', 'scale_height',
                     'r_c', 'flaring_exp', 'PA', 'dust_param', 'vturb',
                     'gas_mass', 'gasdust_ratio']
        elif param == 'co_df':
            param = ['inclination', 'stellar_mass', 'scale_height',
                     'r_c', 'flaring_exp', 'PA', 'dust_param', 'vturb',
                     'gas_mass', 'gasdust_ratio', 'co_abundance', 'depletion_factor']
        else:
            param = param
            self.param = param

        if 'pb_attenuate' not in kwargs:
            kwargs['pb_attenuate'] = False
            
        print('Setting up priors')
        ultra_priors = importlib.import_module('ultranest_priors')
        my_prior_transform = ultra_priors.create_prior_transform(param)
        # priors for caching
        Ndim = len(param)
        cube = np.random.uniform(size=Ndim)
        p0 = [my_prior_transform(cube)]
            
        # Acquire and store the GCF and CORR caches for iterative sampling
        print('Caching...')
        infdata = self.cache(p0, infdata, restfreq, kwargs)

        # Declare the data and kwargs as globals (for speed in pickling)
        global fdata
        global kw
        fdata = copy.deepcopy(infdata)
        kw = copy.deepcopy(kwargs)

        # Populate keywords from kwargs dictionary
        if 'restfreq' not in kw:
            kw['restfreq'] = restfreq
        if 'FOV' not in kw:
            kw['FOV'] = 5.0
        if 'Npix' not in kw:
            kw['Npix'] = 256
        if 'dist' not in kw:
            kw['dist'] = 150.
        if 'cfg_dict' not in kw:
            kw['cfg_dict'] = {}
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


        print('Initialising sampler')
        sampler = ultranest.ReactiveNestedSampler(param, self.log_likelihood_ultranest,
                                                  my_prior_transform, log_dir=outpost,
                                                  resume=resume, run_num=run_num)

        if trace == True:
            sampler.plot_trace()
            return


        print('Running sampler')
        result = sampler.run()
        sampler.print_results()

        return
    
    """
        Function to initialise the priors for Caitlyn's MCFOST models
    """
    def mcfost_priors(self, priors, nwalk, ndim, initial=None):
        if initial is not None:
           fractional_change = 1e-2*np.abs(initial)
           p0 = initial + fractional_change*np.random.randn(nwalk, ndim)
        else:
            p0 = np.empty((nwalk, ndim))
            for ix in range(ndim):
                if priors.pri_types[ix] == "normal" or priors.pri_types[ix] == "uniform":
                    _ = [str(priors.pri_pars[ix][ip])+', ' for ip in range(len(priors.pri_pars[ix]))]
                    cmd = 'np.random.'+priors.pri_types[ix]+'('+"".join(_)+str(nwalk)+')'
                    p0[:,ix] = eval(cmd)
                elif priors.pri_types[ix] == "truncnorm" or priors.pri_types[ix] == "loguniform":
                    if priors.pri_types[ix] == "truncnorm":
                        params = priors.pri_pars[ix]
                        mod_pri_pars = [(params[2]-params[0])/params[1], (params[3]-params[0])/params[1], params[0], params[1]]
                        _ = [str(mod_pri_pars[ip])+', ' for ip in range(len(mod_pri_pars))]
                    else:
                        _ = [str(priors.pri_pars[ix][ip])+', ' for ip in range(len(priors.pri_pars[ix]))]
                    cmd = 'stats.'+priors.pri_types[ix]+'.rvs('+"".join(_)+'size='+str(nwalk)+')'
                    p0[:,ix] = eval(cmd)
                else:
                    raise NameError('Prior type unaccounted for')
        return p0
    

    """
        Function to acquire and store the GCF and CORR caches for iterative sampling
    """
    def cache(self, p0, infdata, restfreq, kwargs):
        icube = None
        for i in range(infdata['Nobs']):
            if self.prescription != 'MCFOST' or i==0:
                icube = None
            _, gcf, corr, _icube, image_lnl = self.modelset(dset=infdata[str(i)], pars=p0[0], 
                                         restfreq=restfreq, 
                                         FOV=kwargs['FOV'],
                                         Npix=kwargs['Npix'], 
                                         dist=kwargs['dist'],
                                         cfg_dict=kwargs['cfg_dict'],
                                         return_holders=True,
                                         icube=copy.deepcopy(icube),
                                         param=self.param, 
                                         attenuate=kwargs['pb_attenuate'])
            if icube is None:
                icube = _icube
            infdata['gcf_'+str(i)] = gcf
            infdata['corr_'+str(i)] = corr
    
        return infdata


    """
        Function to calculate a log-posterior sample.
    """
    def log_posterior(self, theta, model=None):

        # Calculate log-prior
        priors = importlib.import_module('priors_'+self.priors_prescription)
        lnT = np.sum(priors.logprior(theta)) * fdata['Nobs']
        if isnan(lnT):
            print(lnT)
        if lnT == -np.inf:
            return -np.inf, -np.inf

        # Compute log-likelihood
        lnL, image_lnl = self.log_likelihood(theta, fdata=fdata, kwargs=kw, model_vis=model)

        if image_lnl is None:
            # return the log-posterior and the log-prior
            return lnL + lnT, lnT
        else:
            return lnL + lnT, lnT, image_lnl
        
    
    """
       Function to calculate a log-likelihood for ultranest
    """
    def log_likelihood_ultranest(self, theta):

        lnL, image_lnl = self.log_likelihood(theta, fdata=fdata, kwargs=kw)

        return lnL



    """
        Function to calculate a log-likelihood.
    """
    def log_likelihood(self, theta, fdata=None, kwargs=None, optimise=False, model_vis=None, plot_brute=False, my_uvtaper=True):

        # Loop over observations to get likelihood
        logL = 0
        icube = None
        image_lnl = None

        if plot_brute:
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))


        for i in range(fdata['Nobs']):
            #t0 = time.time()

            # Get the data 
            _data = fdata[str(i)]

            # only reuse the model for mcfost run
            if self.prescription != 'MCFOST' or i==0:
                icube = None

            # Calculate the model
            _mdl, _, lnl = self.modelset(_data, theta, restfreq=kwargs['restfreq'],
                                 FOV=kwargs['FOV'], Npix=kwargs['Npix'], 
                                 dist=kwargs['dist'], cfg_dict=kwargs['cfg_dict'],
                                 chpad=kwargs['chpad'],
                                 doppcorr=kwargs['doppcorr'], 
                                 SRF=kwargs['SRF'], 
                                 gcf_holder=fdata['gcf_'+str(i)],
                                 corr_cache=fdata['corr_'+str(i)], 
                                 icube=copy.deepcopy(icube), param=self.param,  
                                 attenuate=kwargs['pb_attenuate'])
            
            #t1 = time.time()
            #print('Time for EB '+str(i)+' is '+str(t1-t0))
        
            if icube is None:
                icube = _
            if image_lnl is None:
                image_lnl = lnl
        
            
            # plot next to each other
            if plot_brute:
                uv_dist = np.sqrt(_data.um**2 + _data.vm**2)
                data_pol1 = _data.vis[0][75]
                data_pol2 = _data.vis[1][75]
                model = _mdl.vis[0][151]
                #dp1_av = np.mean(data_pol1, axis=0)
                #dp2_av = np.mean(data_pol2, axis=0)
                #m_av = np.mean(model, axis=0)
                j = i//3
                k = i % 3
                axs[j][k].scatter(uv_dist, data_pol1, color='red', label='Data Pol 1', s=1, alpha=0.3)
                axs[j][k].scatter(uv_dist, data_pol2, color='purple', label='Data Pol 2', s=1, alpha=0.3)
                axs[j][k].scatter(uv_dist, model, color='blue', label='Model', s=1, alpha=0.3)
                axs[j][k].set_xlabel('UV distance')
                axs[j][k].set_ylabel('Visibility')
                axs[j][k].set_title('EB '+str(i))


            # Spectrally bin the model visibilities if necessary
            # **technically wrong, since the weights are copied like this; 
            # **ideally would propagate the unbinned weights?
            if fdata['chbin'][i] > 1:
                oshp = (_mdl.npol, -1, fdata['chbin'][i], _mdl.nvis)
                wt = np.rollaxis(np.tile(_data.wgt, (2, 1, 1, 1)), 0, 3)
                wt[wt == 0.] = 1    # mitigate normalization issue for censored
                mvis = np.average(_mdl.vis.reshape(oshp), weights=wt, axis=2)
            else:
                mvis = 1. * _mdl.vis

            # Compute the residual and variance matrices(stack both pols)
            if model_vis is None:
                resid = np.hstack(np.absolute(_data.vis - mvis))
                print(resid.shape)
            else:
                resid = np.hstack(np.absolute(model_vis[str(i)] - mvis))

            cfg_dict=kwargs['cfg_dict']

            if cfg_dict.get('uv_taper') is not None:
                # = 0.15 arcsec beam in kλ = 1375
                print('taper')
                uvtaper = cfg_dict.get('uv_taper') # in arcseconds
                uvtaper = 206265/(uvtaper*1000) # in kλ
                print(uvtaper)
                new_weights = _data.wgt*np.exp(-(_data.um**2 + _data.vm**2)/uvtaper**2)
                var = np.hstack(new_weights)
            else:
                var = np.hstack(_data.wgt)


            # Compute the log-likelihood (** still needs constant term)
            Cinv = fdata['invcov_'+str(i)]
            if cfg_dict.get('cov_on') is None or cfg_dict.get('cov_on') is False:
                print('covariance matrix is off')
                Cinv = np.identity(Cinv.shape[0])
            if cfg_dict.get('eb_contribution') is not None:
                print('cov matrix back and lnL0!!')
                constant_term = fdata['lnL0_'+str(i)]
                Cinv = fdata['invcov_'+str(i)]
                print('EB '+str(i)+': '+str(-0.5 * np.tensordot(resid, np.dot(Cinv, var * resid))-constant_term))
                logL += -0.5 * np.tensordot(resid, np.dot(Cinv, var * resid)) - constant_term
            elif cfg_dict.get('manual_scaling') is not None:
                print('manual scaling')
                constant_term = cfg_dict.get('manual_scaling')
                logL += -0.5 * np.tensordot(resid, np.dot(Cinv, var * resid)) - constant_term/fdata['Nobs']

            else:
                logL += -0.5 * np.tensordot(resid, np.dot(Cinv, var * resid))

        if isnan(logL):
            print('logL is NaN: ', theta, flush=True)

        if plot_brute:
                plt.savefig('plotting_vis_data_model.png')
                plt.savefig('plotting_vis_data_model.pdf')

        return logL, image_lnl


def determinant(args):
    ii, jj = args
    print(ii)
    sgn, lndet = np.linalg.slogdet(scov/_wgt[jj,:,ii])
    return (jj, ii, sgn*lndet)

def apply_primary_beam_effect(model_cube, pixel_scale, frequency, diameter):
    """
    Apply primary beam attenuation to a model cube (to simulate ALMA observation).
    
    Parameters:
    - model_image (3D numpy array): The input model cube.
    - pixel_scale (float): Arcseconds per pixel.
    - frequency (float): Frequency in Hz.
    - diameter (float): Telescope diameter in meters (7m and 12m for ALMA).
    
    Returns:
    - attenuated_cube (3D numpy array): Model cube after primary beam attenuation.
    """

    c_m_per_s = 299792458
    wavelength_m = c_m_per_s / frequency
    # Compute primary beam FWHM
    theta_b = (1.13 * wavelength_m / diameter) * (180/np.pi) * 3600  # in arcsec
    # Get image shape
    nx, ny, nchan = model_cube.shape
    # Generate coordinate grid (centered)
    x = np.arange(-nx//2, nx//2) * pixel_scale
    y = np.arange(-ny//2, ny//2) * pixel_scale
    X, Y = np.meshgrid(x, y)
    # Compute radius from center
    r = np.sqrt(X**2 + Y**2)
    # Compute primary beam response
    PB = np.exp(-4 * np.log(2) * (r**2) / (theta_b**2))
    # Apply attenuation (multiply instead of divide, we are doing the opposite of correction)
    attenuated_cube = model_cube * PB[:, :, np.newaxis]

    return attenuated_cube