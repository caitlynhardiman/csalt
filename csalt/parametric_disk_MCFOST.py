import pymcfost as mcfost
import scipy.constants as sc
import numpy as np
from vis_sample.classes import SkyImage
import matplotlib.pyplot as plt
import os
import subprocess
import multiprocess

def parametric_disk(velax, pars, pars_fixed, newcube):

    restfreq, FOV, npix, dist, cfg_dict = pars_fixed  # these need to come in somewhere, right now they are manually in the para file

    model = write_run_mcfost(pars)

    x = model.pixelscale * (np.arange(model.nx) - model.cx +1)
    y = model.pixelscale * (np.arange(model.ny) - model.cy +1)

    im_cube = model.lines[:, :, :]

    # Re-orient cube array
    cube = np.rollaxis(im_cube, 0, 3)
    
    for_csalt = SkyImage(cube, x, y, model.nu, None)

    return for_csalt



def write_run_mcfost(inclination=None, stellar_mass=None, scale_height=None,
                     r_c=None, r_in=None, flaring_exp=None, PA=None, dust_param=None,
                     vturb=None):
    # Rewrite mcfost para file
    pool_id = multiprocess.current_process()
    pool_id = pool_id.pid
    if os.path.isdir(str(pool_id)) == False:
        subprocess.call("mkdir "+str(pool_id), shell = True)
    updating = mcfost.Params('csalt.para')
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
    para = str(pool_id)+'/csalt_'+str(pool_id)+'.para'
    updating.writeto(para)
    origin = os.getcwd()
    os.chdir(str(pool_id))
    mcfost.run('csalt_'+str(pool_id)+'.para', options="-mol -casa -photodissociation", delete_previous=True, logfile='mcfost.log')
    os.chdir(origin)
    model = mcfost.Line(str(pool_id)+'/data_CO/')
    return model
