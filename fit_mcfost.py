import os, sys
from csalt.data import *

# I/O
datafile = ''  # need to fill
post_dir = 'storage/posteriors/mcfost/'
postfile = ''  # need to fill

# model setups
mtype = 'MCFOST'
mode  = 'iter'
#vra_fit = [-1e3, 11e3]    # (LSRK velocities from -1 to +11 km/s)
vra_fit = None
vcensor = None

# inference setups - can change these here
nwalk = 75
ninits = 300
nsteps = 5000
nthreads = 6

# fixed parameters
nu_rest = 345.796e9                                # rest frequency of line (Hz)
FOV = 6.375                                        # field of view (arcsec)
Npix = 256                                         # pixels per side
dist = 144.5                                       # distance (pc)
cfg_dict = {}                                      # empty dict for CSALT models
fixed = nu_rest, FOV, Npix, dist, cfg_dict




# Set up the prior functions
if not os.path.exists('priors_'+mtype+'.py'):
    print('There is no such file "priors_"+mtype+".py".\n')
    sys.exit()
else:
    os.system('rm priors.py')
    os.system('cp priors_'+mtype+'.py priors.py')

# If necessary, make the posteriors directory
if not os.path.exists(post_dir):
    os.system('mkdir '+post_dir)

# Run the inference
from csalt.fit import *
run_emcee(datafile, fixed, code=mtype, vra=vra_fit, vcensor=vcensor, nwalk=nwalk,
          ninits=ninits, nsteps=nsteps, outfile=post_dir+postfile, mode=mode,
          nthreads=nthreads, append=True)
