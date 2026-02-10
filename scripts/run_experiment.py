#!/usr/bin/env python3
"""
Entry Point for Exploration Experiment
"""

import argparse
import pdb
from typing import Optional
import signal

import gin
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.spatial.transform import Rotation

from dair_exploration import mjx_util
from dair_exploration.file_util import copy_run_config, get_config, results_dir
from dair_exploration.gui_util import MJXMeshcatVisualizer
from dair_exploration.trifinger_utils import TrifingerLCMService
from dair_exploration.action_utils import (
    ActionWorkspaceParams,
    ActionCEM,
    action_to_knots,
)

## Handle SIGINT
signal_pressed = False


def signal_handler(_sig, _frame):
    """Handle SIGINT"""
    # pylint: disable=global-statement
    global signal_pressed
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal_pressed = True
    signal.signal(signal.SIGINT, signal_handler)


## Main Function
@gin.configurable
def main(
    config_file: str,
    model_file: str,
):
    """Main function for online learning loop"""

    # pylint: disable=too-many-locals

    # Debug: Remove scientific notation for numpy printing
    np.set_printoptions(suppress=True)
    print("Active Tactile Exploration")

    # Handle SIGINT
    global signal_pressed
    signal_pressed = False
    signal.signal(signal.SIGINT, signal_handler)

    # Create results directory
    run_dir = results_dir()
    # Save config files to run dir
    copy_run_config(get_config(config_file), "config.gin")
    copy_run_config(get_config(model_file), "model.mjcf")
    print(f"Storing data and results at {run_dir}")

    # Initialize LCM
    action_params = ActionWorkspaceParams()
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    trifinger_lcm = TrifingerLCMService()
    print("Resetting Trifinger Position...")
    trifinger_lcm.execute_trajectory(action_params.get_reset_knot(), no_data=True)
    new_trajectory = None

    # Create learnable system
    print("JIT Mujoco XLA Step...")
    mjx_model = mjx.put_model(
        mujoco.MjModel.from_xml_path(get_config(model_file).as_posix())
    )
    mjx_init_data = mjx.make_data(mjx_model)

    # GUI Visualization
    gui_vis = MJXMeshcatVisualizer(
        mjx_model, mjx_util.jit_forward(mjx_model, mjx_init_data)
    )

    # Sample initial action (from true obj pose)
    action_cem = ActionCEM(action_params)
    selected_action = action_params.random_action()
    selected_knots = np.stack(
        [action_params.get_reset_knot(), action_params.get_reset_knot()]
    )
    first_knots = action_to_knots(
        action_params,
        [selected_action],
        trifinger_lcm.get_current_object_pose(),
        force_finger=0,
    )[0]
    # Move both fingers
    selected_knots = first_knots
    # Only move one finger
    # selected_knots[:, :3] = first_knots[:, :3]
    gui_vis.draw_action_samples(selected_knots[np.newaxis, :, :])

    breakpoint()

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "e - Execute selected action + collect data\n"
            "b - breakpoint()\n"
            "h - Print Help\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == "h":
            print_help()

        elif command_char == "b":
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()
            # ipdb.set_trace()

        elif command_char == "e":
            ## Execute selected action
            # Move to start state
            trifinger_lcm.execute_trajectory(selected_action[0], no_data=True)

            # Execute and collect data
            new_trajectory = trifinger_lcm.execute_trajectory(selected_action[1])

            if len(new_trajectory) < 1:
                print("WARNING: No data collected")
                continue

            # Construct control, last timestep control repeats
            qpos_qvel = jnp.hstack(
                [
                    jnp.hstack(
                        [
                            new_trajectory[1][geom_name][value]
                            for geom_name in trifinger_lcm.fingertip_geom_names
                        ]
                    )
                    for value in ["position", "velocity"]
                ]
            )
            ctrl = jnp.vstack([qpos_qvel[1:, :], qpos_qvel[-1:, :]])

            # Overwrite robot qpos/qvel
            qpos_overwrite = {}
            qvel_overwrite = {}
            for geom_name in trifinger_lcm.fingertip_geom_names:
                qpos_overwrite[geom_name] = new_trajectory[1][geom_name]["position"]
                qvel_overwrite[geom_name] = new_trajectory[1][geom_name]["velocity"]

            print("Running simulation...")
            new_data = diffsim.diffsim_overwrite(
                mjx_model, mjx_init_data, ctrl, qpos_overwrite, qvel_overwrite
            )

            print("Visualizing...")
            obj_traj = {
                trifinger_lcm.object_geom_name: [
                    (
                        new_trajectory[1][trifinger_lcm.object_geom_name]["position"][
                            idx, 4:
                        ],
                        Rotation.from_quat(
                            new_trajectory[1][trifinger_lcm.object_geom_name][
                                "position"
                            ][idx, :4],
                            scalar_first=True,
                        ),
                    )
                    for idx in range(ctrl.shape[0])
                ]
            }
            gui_vis.update_visuals(mjx_model, new_data, obj_traj)

    # Quit
    print("Done!")


def main_fn():
    """Entry point"""
    parser = argparse.ArgumentParser(
        prog="run_experiment.py", description="Run the Active Exploration Experiment"
    )
    parser.add_argument(
        "--config_file", default="default.gin", help="GIN config in /config"
    )
    args = parser.parse_args()

    # Parse config file and start
    gin.register(np.array, module="np")
    gin.register(np.random.uniform, module="np.random")
    print(f"Loading Config File: {get_config(args.config_file)}")
    gin.parse_config_file(get_config(args.config_file))
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    main(config_file=args.config_file)


if __name__ == "__main__":
    main_fn()
