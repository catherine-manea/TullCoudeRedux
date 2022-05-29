# TullCoudeRedux
Reduction Script for the Tull Coude Spectrograph on the 2.7m Telescope at McDonald Observatory

File redux.py contains the functions and script to reduce spectral data.

So far, I have written the bias and flat field correction, order extraction, scattered light subtraction, and 1d spectral extraction steps.  Final step is to do the wavelength calibration.

My plan for the wavelength calibration is to run a cross correlation between each order and a wavelength-calibrated ThAr spectrum.  This cross correlation will also need to contain a term for the resolution/dispersion (i.e. the value that turns each pixel into a delta lambda).
