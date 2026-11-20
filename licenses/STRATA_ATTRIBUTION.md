SCREAM-derived static forcing files — CC BY 4.0
===============================================

The following files in this repository are derived from the SCREAM (Simple
Cloud-Resolving E3SM Atmosphere Model) simulation output that NVIDIA published
as the STRATA-SCREAM datasets under the Creative Commons Attribution 4.0
International license (https://creativecommons.org/licenses/by/4.0/; full text
in third_party/CC-BY-4.0.txt):

  phis.npy      surface geopotential (static forcing field)
  sgh30.npy     standard deviation of sub-grid orography (static forcing field)
  landfrac.npy  land fraction (static forcing field)

All other files (model weights, normalization statistics, configuration and model
card) are original NVIDIA works licensed under the OpenMDW License Agreement,
version 1.1 (see LICENSE).

Sources
-------
- STRATA-SCREAM datasets (CC BY 4.0), published by NVIDIA; dataset owner: E3SM Project,
  U.S. Department of Energy. Simulations F20TR-SCREAMv1 (sdecadal) and F2010-SCREAMv1
  (sdy1, sdy2) on the ne1024pg2 cubed-sphere grid.
  https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdecadal
  https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1
  https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy2
  Original raw SCREAM output: https://portal.nersc.gov/archive/home/n/ndk/www/STRATA2026
- SCREAM / E3SM: Caldwell, P. M., et al. (2021), Convection-Permitting Simulations With the
  E3SM Global Atmosphere Model, J. Adv. Model. Earth Syst., https://doi.org/10.1029/2021MS002544

Changes made by NVIDIA: the static fields were extracted from the simulation output,
cast to 32-bit floats and stored as NumPy arrays in the model's ne1024pg2 column
order; no values were modified.
