# DAIRLab Active Tactile Exploration
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

## Installation
Default installation can be done with pip:
```
$ pip install -e .
```

If you plan to contribute code, please additionally install linters.
```
$ pip install -e .[dev]
```

### Additional Solvers
The default QP solver for this project is [mpax](https://github.com/MIT-Lu-Lab/MPAX). We additionally support [jaxopt](https://jaxopt.github.io/stable/) and [moreau](https://www.moreau.so/), which can be installed separately:
```
$ pip install jaxopt
$ pip install moreau[cuda] --extra-index-url https://pypi.fury.io/optimalintellect/
```
Moreau requires a license key and access token (i.e. username for `pypi.fury.io`). For more information, see the [installation documentation](https://docs.moreau.so/installation.html)

## Running the Code
TODO

## API Documentation
TODO

## Attribution notes
TODO

## Mujoco Notes
* Contact normal is +x, points from geom1 to geom2:
```
normals = mjx_data.contact.frame[:, 0, :]
```
* 4D pyramid approximation, see [decodePyramid](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_util_misc.c#L850) 
```
efc_to_cframe (3x4) = [[1., 1., 1., 1.], [mu1, -mu1, 0., 0.], [0., 0., mu2, -mu2]]
force_cframe (3x1) = efc_to_cframe @ pyramid (4x1)
isotropic: mu1 == mu2
force_wframe (3x1) = mjx_data.contact.frame.T @ force_cframe
```
* `[mu1, mu2] = mjx_data.contact.friction[:, :2]`
* `phis = mjx_data.contact.dist`
* `sliding_vels ([3*n_c]x1) = block_diag(efc_to_cframe) @ mjx_data.efc_J @ mjx_data.qvel`
