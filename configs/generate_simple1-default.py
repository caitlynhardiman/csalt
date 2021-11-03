"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""

import numpy as np


# locators
output_dir = 'storage/'
template_dir = output_dir+'obs_templates/'
storage_dir = output_dir+'synth_storage/'
reduced_dir = output_dir+'data/'
casalogs_dir = output_dir+'CASA_logs/'
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# naming
basename = 'simple1-default'
in_MS = storage_dir+basename+'/'+basename
dataname = reduced_dir+basename+'/'+basename


# observation settings
template = ['lmm']
config = ['6']
ttotal = ['20min']
tinteg = ['30s']
date = ['2022/07/15']
HA_0 = ['-0.5h']
RMS = [3.7]             # desired naturally-weighted RMS in mJy/beam/channel


# spectral settings
dnu_native = [122.0703125 * 1e3]	# native channel spacing (Hz)
nu_rest = 230.538e9	# rest frequency (Hz)
V_tune  = [4.0e3]	# LSRK velocity tuning for central channel (m/s)
V_span  = [15.0e3]	# +/- velocity width around vtune for simulation (m/s)
nover = 5	# spectral over-sampling factor (for SRF convolution)


# spatial settings
RA   = '16:00:00.00'	# phase center RA
DEC  = '-40:00:00.00'	# phase center DEC
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


# reduction settings
tavg = ['']
V_bounds = [(5.2 - 10)*1e3, (5.2 + 10)*1e3]
bounds_pad = 3


# Model parameters
incl  = 40.
PA    = 130.
mstar = 0.25
r_l   = 100.
z0    = 3.4
psi   = 1.
T0    = 75.
q     = -0.5
Tmaxb = 20.
sigV0 = 210.
tau0  = 350.
p     = -1.
Vsys  = 0.e3	#5.2e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, z0, psi, T0, q, Tmaxb, 
                  sigV0, tau0, p, Vsys, dx, dy])


# Fixed parameters
FOV  = [6.4]		# full field of view (arcsec)
Npix = [256]  		# number of pixels per FOV
dist = 150.		# distance (pc)

cfg_dict = {}