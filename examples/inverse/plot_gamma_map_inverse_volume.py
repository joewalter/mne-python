"""
===============================================================================
Compute a sparse inverse solution using the Gamma-Map empirical Bayesian method
===============================================================================

References:
----------
.. [1] D. Wipf, S. Nagarajan
   "A unified Bayesian framework for MEG/EEG source imaging",
   Neuroimage, Volume 44, Number 3, pp. 947-966, Feb. 2009.
   DOI: 10.1016/j.neuroimage.2008.02.059
"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

import mne
from mne.datasets import sample
from mne.inverse_sparse import gamma_map, make_stc_from_dipoles
from mne.viz import (plot_dipole_locations, plot_dipole_amplitudes)

print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'

# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.2)

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run the Gamma-MAP method with dipole output
alpha = 0.2
dipoles, residual = gamma_map(
    evoked, forward, cov, alpha, xyz_same_gamma=True, return_residual=True,
    return_as_dipoles=True, loose=1.0, depth=0.8)

###############################################################################
# Plot dipole activations
plot_dipole_amplitudes(dipoles)

# Plot dipole location of the two strongest dipoles with MRI slices
idxs = np.argsort([np.max(np.abs(dip.amplitude)) for dip in dipoles])[-2:]
for idx in idxs:
    plot_dipole_locations(dipoles[idx], forward['mri_head_t'], 'sample',
                          subjects_dir=subjects_dir, mode='orthoview',
                          idx='amplitude')

# # Plot dipole locations of all dipoles with MRI slices
# for dip in dipoles:
#     plot_dipole_locations(dip, forward['mri_head_t'], 'sample',
#                           subjects_dir=subjects_dir, mode='orthoview',
#                           idx='amplitude')

###############################################################################
# Generate stc from dipoles
stc = make_stc_from_dipoles(dipoles, forward['src'])

###############################################################################
# View stc on MRI slices with nilearn
img = mne.save_stc_as_volume(None, stc, forward['src'], mri_resolution=False)
t1_fname = data_path + '/subjects/sample/mri/T1.mgz'

tidx = np.argmin(np.abs(stc.times - 0.082))
plot_stat_map(index_img(img, tidx), t1_fname, threshold=1e-10,
              title='Gamma-MAP (t=%.3f s.)' % stc.times[tidx])
plt.show()
