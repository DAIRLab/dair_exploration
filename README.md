# DAIRLab Active Tactile Exploration
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

## API Documentation
TODO

## Running the Code
TODO

## Attribution notes
TODO

## Mujoco Notes
* Contact normal is +x, points from geom1 to geom2:
```
normals = mjx_data._impl.contact.frame[:, 0, :]
```
* 4D pyramid approximation, see [decodePyramid](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_util_misc.c#L850) 
```
efc_to_cframe (3x4) = [[1., 1., 1., 1.], [mu1, -mu1, 0., 0.], [0., 0., mu2, -mu2]]
force_cframe (3x1) = efc_to_cframe @ pyramid (4x1)
isotropic: mu1 == mu2
force_wframe (3x1) = mjx_data._impl.contact.frame.T @ force_cframe
```
* `[mu1, mu2] = mjx_data._impl.contact.friction[:, :2]`
* `phis = mjx_data._impl.contact.dist`
* `sliding_vels ([3*n_c]x1) = block_diag(efc_to_cframe) @ mjx_data._impl.efc_J @ mjx_data.qvel`
