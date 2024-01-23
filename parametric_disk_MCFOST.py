import pymcfost as mcfost
import scipy.constants as sc
import numpy as np
from vis_sample.classes import SkyImage
import matplotlib.pyplot as plt
import os
import subprocess
import multiprocess

def parametric_disk(velax, pars, pars_fixed, mpi=False):

    restfreq, FOV, npix, dist, cfg_dict, vsyst = pars_fixed  # these need to come in somewhere, right now they are manually in the para file
    
    if isinstance(pars, dict):
        model = write_run_mcfost(velax, mpi=mpi, vsyst=vsyst, **pars)
    else:
        inc, m, h, rc, rin, psi, PA, dust_a, vturb, dust_mass, gasdust_ratio = pars
        model = write_run_mcfost(velax, inc, m, h, rc, rin, psi, PA, dust_a, vturb, dust_mass, gasdust_ratio, vsyst, mpi)

    x = -model.pixelscale * (np.arange(model.nx) - model.cx +1)
    y = model.pixelscale * (np.arange(model.ny) - model.cy +1)

    im_cube = model.lines[:, :, :]

    # Re-orient cube array
    cube = np.rollaxis(im_cube, 0, 3)

    # Account for the fact that RA and x-axis are not defined with the same convention
    # (see comment in classes.py of vis_sample)
    cube = np.fliplr(cube)
    
    for_csalt = SkyImage(cube, x, y, model.nu)

    return for_csalt



def write_run_mcfost(velax, inclination=None, stellar_mass=None, scale_height=None,
                     r_c=None, r_in=None, flaring_exp=None, PA=None, dust_param=None,
                     vturb=None, dust_mass=None, gasdust_ratio=None, vsyst=None, mpi=False, ozstar=False):
    # Rewrite mcfost para file
    pool_id = multiprocess.current_process()
    pool_id = pool_id.pid
    if mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        pool_id = 'rank_'+str(rank)+'_'+str(pool_id)

    if ozstar:
        jobfs = os.getenv("JOBFS")
        directory = jobfs+"/"+str(pool_id)
    else:
        directory = str(pool_id)

    if os.path.isdir(directory) == False:
        subprocess.call("mkdir "+directory, shell = True)
    updating = mcfost.Params('csalt.para')

    updating.mol.molecule[0].nv = len(velax)
    updating.mol.molecule[0].v_min = velax[0]/1000
    updating.mol.molecule[0].v_max = velax[-1]/1000

    if inclination is not None:
        updating.map.RT_imin = inclination+180
        updating.map.RT_imax = inclination+180
    if stellar_mass is not None:
        updating.stars[0].M = stellar_mass
    if scale_height is not None:
        updating.zones[0].h0 = scale_height
    if r_c is not None:
        updating.zones[0].Rc = r_c
    if r_in is not None:
        updating.zones[0].Rin = r_in
    if flaring_exp is not None:
        updating.zones[0].flaring_exp = flaring_exp
    if PA is not None:
        updating.map.PA = PA
    if dust_param is not None:
        updating.simu.viscosity = dust_param
    if vturb is not None:
        updating.mol.v_turb = vturb
    if dust_mass is not None:
        updating.zones[0].dust_mass = dust_mass
    if gasdust_ratio is not None:
        updating.zones[0].gas_to_dust_ratio = gasdust_ratio

    para = directory+'/csalt_'+str(pool_id)+'.para'
    updating.writeto(para)
    origin = os.getcwd()
    os.chdir(directory)
    if vsyst is not None:
        options = "-mol -casa -photodissociation -v_syst " + str(vsyst)
        print(options)
    else:
        options = "-mol -casa -photodissociation"
    mcfost.run('csalt_'+str(pool_id)+'.para', options=options, delete_previous=True, logfile='mcfost.log')
    os.chdir(origin)
    model = mcfost.Line(directory+'/data_CO/')
    return model
