# Changelog (PsyTrack)

## 1.3.0 (April 18, 2020)

- Added a `hess_calc` argument to `recoverSim` to control whether error bars are calculated in the recovery of simulated weights.
- Added a `hess_calc` argument to `crossValidate` to control whether error bars are calculated in the recovery of simulated weights.
- Change display of error bars from 2 standard errors, to 1.96 which maps more precisely to a 95% credible interval.
- Update the analysis function to handle `s_a` and `s_b` as inputs.
- Update the analysis function to better plot session boundaries.
- Update the analysis function to handle Left/Right sides coded as {0,1}, as well as {1,2}.
- The default z-order (plotting order) of different inputs can now be accessed through `psy.ZORDER`, similar to `psy.COLORS`.
- Made a few superficial updates to the tutorial notebook.


## 1.2.0 (February 16, 2020)

- This may not be a backward compatible update, apologies in advance!
- Added functionality to `hyperOpt` for returning error bars for the weights returned in `best_wMode` by setting the keyword argument `hess_calc`.
- Added functionality to `hyperOpt` for returning error bars for the hyperparameters returned in `best_hyper` by setting the keyword argument `hess_calc`.
- Added check to `getMap` raise error if choices `y` are not formatted correctly.
- Corrected a minor zero-indexing bug in the cross-validation code.
- Adjusted `runSim` code to simulate and recover models using the `sigDay` functionality.
- Update documentation to clarify that the order of weights returned in `wMode` is alphabetical with respect to the dictionary `weights`.
- Adjust the `__init__.py` file to make commonly used functions more directly accessible.
- Large improvements in the plotting functions, including more modularity and configurability.
- Made cross-validation much more straightforward, should be able to get log-likelihood direct from a single function now.
- Refresh of tutorial notebook

## 1.1.0 (June 18, 2019)

- Rename `aux` folder to `helper` to allow for installation in Windows systems