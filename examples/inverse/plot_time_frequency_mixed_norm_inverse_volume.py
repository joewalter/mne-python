"""
=============================================
Compute MxNE with time-frequency sparse prior
=============================================

The TF-MxNE solver is a distributed inverse method (like dSPM or sLORETA)
that promotes focal (sparse) sources (such as dipole fitting techniques).
The benefit of this approach is that:

  - it is spatio-temporal without assuming stationarity (sources properties
    can vary over time)
  - activations are localized in space, time and frequency in one step.
  - with a built-in filtering process based on a short time Fourier
    transform (STFT), data does not need to be low passed (just high pass
    to make the signals zero mean).
  - the solver solves a convex optimization problem, hence cannot be
    trapped in local minima.

References:
----------
.. [1] A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
   "Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
   non-stationary source activations",
   Neuroimage, Volume 70, pp. 410-422, 15 April 2013.
   DOI: 10.1016/j.neuroimage.2012.12.051

.. [2] A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
   "Functional Brain Imaging with M/EEG Using Structured Sparsity in
   Time-Frequency Dictionaries",
   Proceedings Information Processing in Medical Imaging
   Lecture Notes in Computer Science, Volume 6801/2011, pp. 600-611, 2011.
   DOI: 10.1007/978-3-642-22092-0_49
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
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.inverse_sparse import tf_mixed_norm, make_stc_from_dipoles
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
evoked.crop(tmin=-0.05, tmax=0.25)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run solver
# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=None, depth=None)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute TF-MxNE inverse solution with dipole output
dipoles, residual = tf_mixed_norm(
    evoked, forward, cov, 40., 1., loose=1.0, depth=0.8,
    maxit=1000, tol=1e-6, weights=stc_dspm, weights_min=8., debias=True,
    wsize=16, tstep=4, window=0.05, return_as_dipoles=True,
    return_residual=True)

# Crop to remove edges
for dip in dipoles:
    dip.crop(tmin=0.0, tmax=0.2)
evoked.crop(tmin=0.0, tmax=0.2)
residual.crop(tmin=0.0, tmax=0.2)

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

tidx = np.argmin(np.abs(stc.times - 0.101))
plot_stat_map(index_img(img, tidx), t1_fname, threshold=1e-10,
              title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])
plt.show()
