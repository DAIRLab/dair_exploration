#!/usr/bin/env python3
"""
Entry Point for Exploration Experiment
"""

import argparse
import pdb
from typing import Optional

import gin
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.spatial.transform import Rotation

from dair_exploration import diffsim
from dair_exploration.file_utils import get_config
from dair_exploration.gui_utils import MJXMeshcatVisualizer
from dair_exploration.trifinger_utils import (
    TrifingerLCMService,
    sample_action,
    Action,
)


## Main Function
@gin.configurable
def main(
    model_file: str,
    action_library: Optional[list[Action]] = None,
):
    """Main function for online learning loop"""
    # Debug: Remove scientific notation for numpy printing
    np.set_printoptions(suppress=True)
    print("Active Tactile Exploration")

    # Create learnable system
    print("JIT Mujoco XLA Step...")
    mjx_model = mjx.put_model(mujoco.MjModel.from_xml_path(get_config(model_file)))
    mjx_init_data = mjx.make_data(mjx_model)

    # GUI Visualization
    gui_vis = MJXMeshcatVisualizer(
        mjx_model, diffsim.jit_step(mjx_model, mjx_init_data)
    )

    # Initialize LCM
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    trifinger_lcm = TrifingerLCMService()
    print("Sample Initial Random Action...")
    selected_action = sample_action(library=action_library)
    new_trajectory = None

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
    print(f"Loading Config File: {get_config(args.config_file)}")
    gin.parse_config_file(get_config(args.config_file))
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    main()


if __name__ == "__main__":
    main_fn()
