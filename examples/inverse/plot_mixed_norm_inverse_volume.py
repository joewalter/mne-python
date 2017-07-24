"""
================================================================
Compute sparse inverse solution with mixed norm: MxNE and irMxNE
================================================================

Runs (ir)MxNE (L1/L2 [1]_ or L0.5/L2 [2]_ mixed norm) inverse solver.
L0.5/L2 is done with irMxNE which allows for sparser
source estimates with less amplitude bias due to the non-convexity
of the L0.5/L2 mixed norm penalty.

References
----------
.. [1] Gramfort A., Kowalski M. and Hamalainen, M.
   "Mixed-norm estimates for the M/EEG inverse problem using accelerated
   gradient methods", Physics in Medicine and Biology, 2012.
   http://dx.doi.org/10.1088/0031-9155/57/7/1937.

.. [2] Strohmeier D., Haueisen J., and Gramfort A.
   "Improved MEG/EEG source localization with reweighted mixed-norms",
   4th International Workshop on Pattern Recognition in Neuroimaging,
   Tuebingen, 2014. 10.1109/PRNI.2014.6858545
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

import mne
from mne.datasets import sample
from mne.inverse_sparse import mixed_norm, make_stc_from_dipoles
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import (plot_dipole_locations, plot_dipole_amplitudes)

print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)
# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.2)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run solver
alpha = 50  # regularization parameter between 0 and 100 (100 is high)
n_mxne_iter = 10  # if > 1 use L0.5/L2 reweighted mixed norm solver
# if n_mxne_iter > 1 dSPM weighting can be avoided.

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=None, depth=None, fixed=False)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute (ir)MxNE inverse solution with dipole output
dipoles, residual = mixed_norm(
    evoked, forward, cov, alpha, loose=1.0, depth=0.8, maxit=3000,
    tol=1e-6, active_set_size=10, debias=True, weights=stc_dspm,
    weights_min=8., n_mxne_iter=n_mxne_iter, return_residual=True,
    return_as_dipoles=True)

###############################################################################
# Plot dipole activations
plot_dipole_amplitudes(dipoles)

# Plot dipole locations of all dipoles with MRI slices
for dip in dipoles:
    plot_dipole_locations(dip, forward['mri_head_t'], 'sample',
                          subjects_dir=subjects_dir, mode='orthoview',
                          idx='amplitude')

###############################################################################
# Generate stc from dipoles
stc = make_stc_from_dipoles(dipoles, forward['src'])

###############################################################################
# View stc on MRI slices with nilearn
img = mne.save_stc_as_volume(None, stc, forward['src'], mri_resolution=False)
t1_fname = data_path + '/subjects/sample/mri/T1.mgz'

tidx = np.argmin(np.abs(stc.times - 0.101))
plot_stat_map(index_img(img, tidx), t1_fname, threshold=1e-10,
              title='MxNE (t=%.3f s.)' % stc.times[tidx])
plt.show()
