#!/usr/bin/env python3

"""
Test Diffsim with overwrite gradients
"""

import jax
import jax.numpy as jnp
from mujoco import mjx

from dair_exploration import mjx_util
from dair_exploration.file_util import enable_jax_cache
from dair_exploration.gui_util import debug_view_simulation
from dair_exploration.learning import LearnedModel


def final_pose_from_initial_pose(
    init_qpos: jax.Array,
    model: mjx.Model,
    ctrl: jax.Array,
    name="object-geom",
):
    """Return final pose given initial pose"""

    start_data = mjx_util.write_qpos_to_data(
        model, mjx.make_data(model), {name: init_qpos}
    )

    step_data = mjx_util.diffsim(model, start_data, ctrl, stacked=True)

    return (
        mjx_util.extract_geom_qposvel_from_data(model, step_data, (name,))[name][
            "position"
        ][-1, ...],
        step_data,
    )


grad_final_pose = jax.jit(jax.jacfwd(final_pose_from_initial_pose, has_aux=True))


def final_pose_with_overwrite(
    init_qpos: jax.Array,
    model: mjx.Model,
    ctrl: jax.Array,
    measurements: dict[str, dict[str, jax.Array]],
    name="object-geom",
    keep_grad: bool = False,
):
    """Return final pose given initial pose"""

    start_data = mjx_util.write_qpos_to_data(
        model, mjx.make_data(model), {name: init_qpos}
    )

    step_data = mjx_util.diffsim_overwrite(
        model, start_data, ctrl, measurements, stacked=True, keep_grad=keep_grad
    )

    return (
        mjx_util.extract_geom_qposvel_from_data(model, step_data, (name,))[name][
            "position"
        ][-1, ...],
        step_data,
    )


grad_final_pose_overwrite = jax.jit(
    jax.jacfwd(final_pose_with_overwrite, has_aux=True),
    static_argnames=("name", "keep_grad"),
)


def test_diffsim_overwrite():
    """Compare gradient of diffsim to gradient of diffsim + overwrite"""

    enable_jax_cache()
    learned_model = LearnedModel("default.xml", {"object-geom": ["size"]})
    mjx_model = learned_model.active_model
    name = "object-geom"
    debug_vis = False

    init_qpos = mjx_util.extract_geom_qposvel_from_data(
        mjx_model, mjx.make_data(mjx_model), (name,)
    )[name]["position"]

    ctrl = jnp.repeat(
        mjx.make_data(mjx_model)
        .ctrl.at[:6]
        .set(jnp.tile(init_qpos[:3], 2))[jnp.newaxis, :],
        30,
        axis=0,
    )

    # Regular diffsim
    print(f"Init Pose (pos, quat scalar first): {jnp.round(init_qpos, 3)}")
    print("Running vanilla diffsim...")
    vanilla, vanilla_data = final_pose_from_initial_pose(init_qpos, mjx_model, ctrl)

    if debug_vis:
        print("Debug in Meshcat...")
        debug_view_simulation(mjx_model, vanilla_data)

    print("Differentiating vanilla diffsim...")
    vanilla_jac, vanilla_grad_data = grad_final_pose(init_qpos, mjx_model, ctrl)

    print(f"Final Pose: {jnp.round(vanilla, 3)}")

    print(
        f"Vanilla Jacobian Diag, should be [~1, ~1, ~0, 0, ~0, ~0, ~1]: {jnp.round(jnp.diag(vanilla_jac), 2)}"
    )
    try:
        assert jnp.all(
            jnp.isclose(vanilla_data.qpos, vanilla_grad_data.qpos)
        ), "Data disagrees"
    except AssertionError as error:
        print(error)

    # Diffsim with overwrite
    robot_names = ("spherebot1-geom", "spherebot2-geom")

    measurements = mjx_util.extract_geom_qposvel_from_data(
        mjx_model, vanilla_data, robot_names + (name,)
    )

    print("Running overwrite diffsim (no keep_grad)...")
    overwrite, overwrite_data = final_pose_with_overwrite(
        init_qpos, mjx_model, ctrl, measurements, keep_grad=False
    )

    if debug_vis:
        print("Debug in Meshcat...")
        debug_view_simulation(mjx_model, overwrite_data)

    print("Differentiating overwrite diffsim...")
    overwrite_jac, overwrite_grad_data = grad_final_pose_overwrite(
        init_qpos, mjx_model, ctrl, measurements, keep_grad=False
    )

    print(f"Final Pose: {jnp.round(overwrite, 3)}")

    print(
        f"Overwrite Jacobian Diag, should be 0s: {jnp.round(jnp.diag(overwrite_jac), 2)}"
    )
    try:
        assert jnp.all(
            jnp.isclose(overwrite_data.qpos, overwrite_grad_data.qpos)
        ), "Data disagrees"
        assert jnp.all(
            jnp.isclose(overwrite_data.qpos, vanilla_data.qpos)
        ), "Vanilla and Overwrite disagree"
    except AssertionError as error:
        print(error)

    # Add keep_grad
    print("Running keep_grad diffsim...")
    keepgrad, keepgrad_data = final_pose_with_overwrite(
        init_qpos, mjx_model, ctrl, measurements, keep_grad=True
    )

    if debug_vis:
        print("Debug in Meshcat...")
        debug_view_simulation(mjx_model, keepgrad_data)

    print("Differentiating keepgrad diffsim...")
    keepgrad_jac, keepgrad_grad_data = grad_final_pose_overwrite(
        init_qpos, mjx_model, ctrl, measurements, keep_grad=True
    )

    print(f"Final Pose: {jnp.round(keepgrad, 3)}")

    print(
        f"Keepgrad Jacobian Diag, should be [~1, ~1, ~0, 0, ~0, ~0, ~1]: {jnp.round(jnp.diag(keepgrad_jac), 2)}"
    )
    try:
        assert jnp.all(
            jnp.isclose(keepgrad_data.qpos, keepgrad_grad_data.qpos)
        ), "Data disagrees"
        assert jnp.all(
            jnp.isclose(keepgrad_data.qpos, vanilla_data.qpos)
        ), "Vanilla and keepgrad disagree"
    except AssertionError as error:
        print(error)

    print("Done!")


if __name__ == "__main__":
    test_diffsim_overwrite()
