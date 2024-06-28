import pymcfost as mcfost
import scipy.constants as sc
import numpy as np
from vis_sample.classes import SkyImage
import matplotlib.pyplot as plt
import os
import subprocess
import multiprocess

def parametric_disk(velax, pars, pars_fixed, mpi=False):

    restfreq, FOV, npix, dist, cfg_dict = pars_fixed  # these need to come in somewhere, right now they are manually in the para file

    # Unpacking options from cfg dict
    ozstar = cfg_dict.get('ozstar')
    vsyst = cfg_dict.get('vsyst')
    directory = cfg_dict.get('directory')
    param = cfg_dict.get('param')
    image_plane_likelihood = cfg_dict.get('image_plane')
    line_profile_likelihood = cfg_dict.get('line_profile')

    if param is not None:
    #param_dict_form = {param: pars[0]}
        param_dict_form = {}
        for i in range(len(param)):
            param_dict_form[param[i]] = pars[i]
        pars = param_dict_form
        model = write_run_mcfost(velax, mpi=mpi, vsyst=vsyst, storage=pars, **pars)       
    else:
        inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio = pars
        model = write_run_mcfost(velax, inc, m, h, rc, psi, PA, dust_a, vturb, gas_mass, gasdust_ratio, vsyst, mpi)

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
                     r_c=None, flaring_exp=None, PA=None, dust_param=None,
                     vturb=None, gas_mass=None, gasdust_ratio=None,vsyst=None, mpi=False, ozstar=False,
                     storage=None):
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
        if storage is not None:
            directory = ''
            params_dir = ''
            for key, value in storage.items():
                directory += key+'_'
                params_dir += str(value)+'_'
            directory = directory[:-1]
            params_dir = params_dir[:-1]
            directory = directory+'/'+params_dir
        else:
            directory = str(pool_id)

    if os.path.isdir(str(pool_id)) == False:
        subprocess.call("mkdir -p "+str(pool_id), shell = True)
    updating = mcfost.Params('csalt.para')

    updating.mol.molecule[0].nv = len(velax)
    updating.mol.molecule[0].v_min = velax[0]/1000
    updating.mol.molecule[0].v_max = velax[-1]/1000

    if inclination is not None:
        updating.map.RT_imin = 180-inclination
        updating.map.RT_imax = 180-inclination
    if stellar_mass is not None:
        updating.stars[0].M = stellar_mass
    if scale_height is not None:
        updating.zones[0].h0 = scale_height
    if r_c is not None:
        updating.zones[0].Rc = r_c
    if flaring_exp is not None:
        updating.zones[0].flaring_exp = flaring_exp
    if PA is not None:
        updating.map.PA = PA+180
    if dust_param is not None:
        updating.simu.viscosity = dust_param
    if vturb is not None:
        updating.mol.v_turb = vturb
    if gasdust_ratio is not None:
        updating.zones[0].gas_to_dust_ratio = gasdust_ratio
    if gas_mass is not None:
        updating.zones[0].dust_mass = gas_mass/updating.zones[0].gas_to_dust_ratio

    if storage is None:
        para = directory+'/csalt_'+str(pool_id)+'.para'
    else:
        para = directory+'/csalt.para'
    updating.writeto(para)
    origin = os.getcwd()
    os.chdir(directory)
    if vsyst is not None:
        print('vsyst = ', vsyst)
        options = "-mol -casa -photodissociation -v_syst " + str(vsyst)
    else:
        print('no vsyst')
        options = "-mol -casa -photodissociation"
    if storage is None:
        mcfost.run('csalt_'+str(pool_id)+'.para', options=options, delete_previous=True, logfile='mcfost.log')
    else:
        mcfost.run('csalt.para', options=options, delete_previous=True, logfile='mcfost.log')
    os.chdir(origin)
    model = mcfost.Line(directory+'/data_CO/')

    return model
