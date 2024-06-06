import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *
from csalt.fit_mcfost import *

data = '../brute_force_visibility/full/lsrk.ms'
directory = 'best_bilby'

# Read in the data MS
ddict = read_MS(data)

# Instantiate a csalt model
cm = model('MCFOST')

# Define some fixed attributes for the modeling
fixed_kw = {'FOV': 6.375, 'Npix': 256, 'dist': 144.5, 'restfreq':345.796e9, 'vsyst':6.04} 

# Set the CSALT model parameters

# image plane
#pars = np.array([36.6, 0.4, 17.5, 250, 7, 1.3, 155, 1e-4, 0.0, 0.0004, 100])

# vis fit
#pars = np.array([36.0, 0.59, 7.2, 160.4, 0.1, 1.665, 158.1, 1e-4, 0.198, 0.042, 925.6])

# vis fit model
# pars = np.array([3.61813656e+01, 5.90392034e-01, 7.82342727e+00, 1.60013840e+02,
#         1.01214382e-01, 1.82646777e+00, 1.55906205e+02, 7.12980994e-04,
#         2.16655923e-01, 3.10952031e-02, 9.95033764e+02])

# 2nd best
# pars = np.array([3.60121609e+01, 5.77353021e-01, 7.31735744e+00, 1.60131193e+02,
#                 1.12728346e-01, 1.78249181e+00, 1.56735250e+02, 5.26370408e-04,
#                 2.08209515e-01, 3.81245791e-02, 9.78413213e+02])

# # worst
# pars = np.array([3.58765963e+01, 5.71402964e-01, 1.35691480e+01, 2.06466352e+02,
#                  1.59861580e+00, 1.29162692e+00, 1.57365851e+02, 1.98641302e-04,
#                  1.20847165e-01, 1.31460026e-02, 4.39554857e+01])

# bilby
pars = np.array([3.09429697e+01,  2.96161988e-01,  7.37481899e+00,
   1.95919981e+02,  5.57429967e+00,  1.24948857e+00, 1.55274210e+02,
   8.10684011e-04,  6.32312351e-04,  9.78569738e-04,  9.97867797e+02])



# flaherty vals
par_names = ['Inc', 'Mstar', 'h', 'r_c', 'r_in', 'psi', 'PA', 'Dust α', 'Vturb', 'Mdust', 'gd ratio']
prec = [0.1, 0.01, 0.1, 0.1, 0.1, 0.001, 0.1, 0.0001, 0.001, 0.000001, 0.1]
#pars = np.array([36.0, 0.54, 18.6, 278, 8, 1.315, 154.8, 1e-4, 0.07, 4e-5, 1000])
#pars = np.array([36.0, 0.54, 18.6, 278, 8, 1.315, 154.8, 1e-4, 0.0, 4e-5, 1000])


# # Calculate a model dictionary; write it to model and residual MS files
print("Model Dictionary")
mdict = cm.modeldict(ddict, pars, kwargs=fixed_kw)
write_MS(mdict, outfile=directory+'/DMTau_MODEL.ms')
write_MS(mdict, outfile=directory+'/DMTau_RESID.ms', resid=True)


# # Define some tclean keywords to be used in all imaging
tclean_kw = {'imsize': 1500, 'start': '4.5km/s', 'width': '0.5km/s',
             'nchan': 7, 'restfreq': '345.796GHz', 'cell': '0.01arcsec',
             'scales': [0, 10, 30, 60], 'niter': 100,
             'robust': 1.0, 'threshold': '3mJy', 'uvtaper': '0.04arcsec'}

# # Define some Keplerian mask keywords
kepmask_kw = {'inc': 36.6, 'PA': 155, 'mstar': 0.4, 'dist': 144.5, 'vlsr': 6100,
              'r_max': 1.1, 'nbeams': 1.5, 'zr': 0.2}

# Image the data, model, and residual cubes

print("Imaging cubes")

imagecube(data, directory+'/DMTau_DATA', 
          kepmask_kwargs=kepmask_kw, tclean_kwargs=tclean_kw)

#tclean_kw['mask'] = 'testdata/DMTau_EB3_DATA.mask'
imagecube(directory+'/DMTau_MODEL.ms', directory+'/DMTau_MODEL', 
          mk_kepmask=False, tclean_kwargs=tclean_kw)
imagecube(directory+'/DMTau_RESID.ms', directory+'/DMTau_RESID',
          mk_kepmask=False, tclean_kwargs=tclean_kw)


# Calculate the likelihood of this model

m = setup_fit(msfile='../exoALMA_ms_files/DM_Tau_ACASBLB_no_ave_selfcal_time_ave_12CO.ms')

infdata = m.initialise()
likelihood = m.get_probability(infdata, pars)
print(likelihood)

### Show the results!

cubes = [directory+'/DMTau_DATA', directory+'/DMTau_MODEL', directory+'/DMTau_RESID']
lbls = ['data', 'model', 'residual']

# Export the cubes to FITS format
from casatasks import exportfits
for i in range(len(cubes)):
    exportfits(cubes[i]+'.image', fitsimage=cubes[i]+'.fits', 
               velocity=True, overwrite=True) 


# Plot a subset of the cube as a direct comparison
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.io import fits
import scipy.constants as sc
#_ = importlib.import_module('plot_setups')
#plt.style.use(['default', '/home/sandrews/mpl_styles/nice_img.mplstyle'])
import cmasher as cmr

nchans = 7

fig, axs = plt.subplots(nrows=3, ncols=nchans, figsize=(9, 3.5))
fl, fr, fb, ft, hs, ws = 0.22, 0.9, 0.12, 0.905, 0.12, 0.03
xlims = [2.5, -2.5]
ylims = [-2.5, 2.5]

for ia in range(len(cubes)):

    print('new cube')

    # Load the cube and header data
    hdu = fits.open(cubes[ia]+'.fits')
    Ico, h = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # coordinate grids, beam parameters
    dx = 3600 * h['CDELT1'] * (np.arange(h['NAXIS1']) - (h['CRPIX1'] - 1))
    dy = 3600 * h['CDELT2'] * (np.arange(h['NAXIS2']) - (h['CRPIX2'] - 1))
    ext = (dx.max(), dx.min(), dy.min(), dy.max())
    bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
    bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi / 180)**2

    # spectral information
    vv = h['CRVAL3'] + h['CDELT3'] * (np.arange(h['NAXIS3']) - (h['CRPIX3']-1))
    print(vv)
    ff = 345.796e9 * (1 - vv / sc.c)

    for i in range(nchans):

        print(i)

        # in-cube index
        j = i + 0

        # convert intensities to brightness temperatures
        Tb = (1e-26 * np.squeeze(Ico[j,:,:]) / bm) * sc.c**2 / \
             (2 * sc.k * ff[j]**2)

        # allocate the panel
        ax = axs[ia, i]
        pax = ax.get_position()

        # plot the channel map
        vmin = 0
        vmax = 80
        cmap = 'cmr.cosmic'
        if ia == 2:
            vmin = -vmax
            cmap = 'cmr.redshift'
        im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=ext,
                       aspect='auto', vmin=vmin, vmax=vmax)

        # set map boundaries
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # overlay beam dimensions
        beam = Ellipse((xlims[0] + 0.1*np.diff(xlims),
                        -xlims[0] - 0.1*np.diff(xlims)), 
                       3600 * bmaj, 3600 * bmin, angle=90-bpa)
        beam.set_facecolor('w')
        ax.add_artist(beam)

        # labeling
        if i == 0:
            ax.text(0.02, 0.90, lbls[ia], transform=ax.transAxes, ha='left',
                    va='center', style='italic', fontsize=8, color='w')
        if ia == 2:
            if np.abs(vv[j]) < 0.001: vv[j] = 0.0
            if np.logical_or(np.sign(vv[j]) == 1, np.sign(vv[j]) == 0):
                pref = '+'
            else:
                pref = ''
            vstr = pref+'%.2f' % (1e-3 * vv[j])
            ax.text(0.97, 0.08, vstr, transform=ax.transAxes, ha='right',
                    va='center', fontsize=7, color='w')
        if np.logical_and(ia == 2, i == 0):
            ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
            ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        else:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

# colorbar
cbax = fig.add_axes([fr+0.01, fb+0.01, 0.02, ft-fb-0.02])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=13)

my_text = '- Parameters -'

for i in range(len(pars)):
    dec_places = np.abs(int(np.log10(prec[i])))
    if dec_places >= 4:
        my_text += '\n' + par_names[i] + ' = ' + ('{:.'+str(1)+'e}').format(pars[i])
    else:
        my_text += '\n' + par_names[i] + ' = ' + ('{:.'+str(dec_places)+'f}').format(pars[i])

props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
axs[0][0].text(-1.98, 0.2, my_text, transform=axs[0][0].transAxes, fontsize=9, horizontalalignment='left', verticalalignment='center', bbox=props)
plt.tight_layout()

fig.suptitle('Likelihood = ' + str(likelihood))
fig.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, hspace=hs, wspace=ws)
fig.savefig(directory+'/model_comparison.pdf')
#fig.savefig('testing_params.png', dpi=300)
fig.clf()
