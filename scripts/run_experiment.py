#!/usr/bin/env python3
"""
Entry Point for Exploration Experiment
"""

import argparse
import pdb
import signal
import time
from typing import Optional

import gin
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.spatial.transform import Rotation

from dair_exploration import mjx_util
from dair_exploration.file_util import (
    enable_jax_cache,
    copy_run_config,
    get_config,
    results_dir,
)
from dair_exploration.gui_util import MJXMeshcatVisualizer
from dair_exploration.trifinger_utils import TrifingerLCMService
from dair_exploration.action_utils import (
    ActionWorkspaceParams,
    ActionCEM,
    action_to_knots,
    interpolate_knots,
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
    enable_jax_cache()
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

    # Sample initial action (from true obj pose)
    action_cem = ActionCEM(action_params)
    selected_action = action_params.random_action()
    selected_knots = np.stack(
        [action_params.get_reset_knot(), action_params.get_reset_knot()]
    )
    start_true_object_pose = trifinger_lcm.get_current_object_pose()
    first_knots = action_to_knots(
        action_params,
        [selected_action],
        start_true_object_pose,
        force_finger=0,
    )[0]
    # Move both fingers
    selected_knots = first_knots
    # Only move one finger
    # selected_knots[:, :3] = first_knots[:, :3]

    # GUI Visualization
    gui_vis = MJXMeshcatVisualizer(
        mjx_model, mjx_util.jit_forward(mjx_model, mjx_init_data)
    )
    gui_vis.update_visuals(
        mjx_model,
        [mjx_util.jit_forward(mjx_model, mjx_init_data)],
        {
            trifinger_lcm.object_geom_name: [
                (
                    start_true_object_pose[4:],
                    Rotation.from_quat(start_true_object_pose[:4], scalar_first=True),
                )
            ]
        },
    )
    gui_vis.draw_action_samples(selected_knots[np.newaxis, :, :])

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
            ## Print Help
            print_help()
            ## END Print Help

        elif command_char == "b":
            ## Debug Breakpoint
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()
            # ipdb.set_trace()
            ## END Debug Breakpoint

        elif command_char == "e":
            ## Collect New Data
            new_trajectory = None
            while new_trajectory is None:
                trifinger_lcm.execute_trajectory(selected_knots[0], no_data=True)
                time.sleep(0.1)

                # Execute and collect data
                new_trajectory = trifinger_lcm.execute_trajectory(selected_knots[1])

                # Move back to start state
                trifinger_lcm.execute_trajectory(selected_knots[0], no_data=True)

                if new_trajectory is None:
                    input("None trajectory, check densetacts. Enter to retry...")

            # Write ctrl to new trajectory
            ctrl_total = interpolate_knots(
                jnp.array(selected_knots), new_trajectory["time"]
            )
            n_q = ctrl_total.shape[-1] // 2
            for geom_idx, geom_name in enumerate(trifinger_lcm.fingertip_geom_names):
                new_trajectory[geom_name]["ctrl"] = np.concatenate(
                    [
                        ctrl_total[:, geom_idx * 3 : (geom_idx + 1) * 3],
                        ctrl_total[
                            :, (n_q + geom_idx * 3) : (n_q + (geom_idx + 1) * 3)
                        ],
                    ],
                    axis=-1,
                )

            # Write data to TrajectorySet

            # Visualize New Data
            gui_vis.update_visuals(
                mjx_model,
                [mjx_util.jit_forward(mjx_model, mjx_init_data)]
                * len(new_trajectory["time"]),
                {
                    trifinger_lcm.object_geom_name: [
                        (
                            row[4:],
                            Rotation.from_quat(row[:4], scalar_first=True),
                        )
                        for row in new_trajectory[trifinger_lcm.object_geom_name][
                            "position"
                        ]
                    ],
                }
                | {
                    geom_name: [
                        (
                            row,
                            Rotation.identity(),
                        )
                        for row in new_trajectory[geom_name]["position"]
                    ]
                    for geom_name in trifinger_lcm.fingertip_geom_names
                },
            )

            ## END Collect New Data

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
