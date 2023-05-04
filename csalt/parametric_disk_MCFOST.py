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
    inc, mass, h, rc, rin, psi, pa, dust_param, vturb = pars

    model = write_run_mcfost(inc, mass, h, rc, rin, psi, pa, dust_param, vturb)

    x = model.pixelscale * (np.arange(model.nx) - model.cx +1)
    y = model.pixelscale * (np.arange(model.ny) - model.cy +1)

    cube = model.lines[:, :, :]

    for_csalt = SkyImage(np.transpose(cube), x, y, model.nu, None)

    return for_csalt



def write_run_mcfost(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, vturb):
    # Rewrite mcfost para file
    pool_id = multiprocess.current_process()
    pool_id = pool_id.pid
    if os.path.isdir(str(pool_id)) == False:
        subprocess.call("mkdir "+str(pool_id), shell = True)
    #print(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, vturb)
    updating = mcfost.Params('dmtau.para')
    updating.map.RT_imin = inclination+180
    updating.map.RT_imax = inclination+180
    updating.stars[0].M = stellar_mass
    updating.zones[0].h0 = scale_height
    updating.zones[0].Rc = r_c
    updating.zones[0].Rin = r_in
    updating.zones[0].flaring_exp = flaring_exp
    updating.map.PA = PA
    updating.simu.viscosity = dust_param
    updating.mol.v_turb = vturb
    para = str(pool_id)+'/dmtau_'+str(pool_id)+'.para'
    updating.writeto(para)
    origin = os.getcwd()
    os.chdir(str(pool_id))
    mcfost.run('dmtau_'+str(pool_id)+'.para', options="-mol -casa -photodissociation", delete_previous=True, logfile='mcfost.log')
    os.chdir(origin)
    model = mcfost.Line(str(pool_id)+'/data_CO/')
    return model
