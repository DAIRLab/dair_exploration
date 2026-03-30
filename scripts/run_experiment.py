#!/usr/bin/env python3
"""
Entry Point for Exploration Experiment
"""

import argparse
import pdb
import pickle
import signal
import time

import gin
import jax.numpy as jnp
from mujoco import mjx
import numpy as np
import optax
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
    action_to_knots,
    interpolate_knots,
)
from dair_exploration.data_util import TrajectorySet
from dair_exploration.learning import LearnedTrajectory, LearnedModel, train_epochs
from dair_exploration.exploration import (
    observed_info,
    InfoHyperparameters,
    expected_info,
)
from dair_exploration.solvers import configure_solvers


## Main Function
@gin.configurable
def main(
    config_file: str,
    model_file: str,
    start_with_true_object_pose: bool = True,
) -> None:
    """Main function for online learning loop"""

    # pylint: disable=too-many-locals,too-many-statements

    # Debug: Remove scientific notation for numpy printing
    np.set_printoptions(suppress=True)
    enable_jax_cache()
    print("Active Tactile Exploration")

    # Handle SIGINT
    signal_pressed = False

    def signal_handler(_sig, _frame):
        nonlocal signal_pressed
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal_pressed = True
        signal.signal(signal.SIGINT, signal_handler)

    signal.signal(signal.SIGINT, signal_handler)

    # Create results directory
    run_dir = results_dir()
    # Save config files to run dir
    copy_run_config(get_config(config_file), "config.gin")
    copy_run_config(get_config(model_file), "model.xml")
    print(f"Storing data and results at {run_dir}")

    # Initialize LCM
    action_params = ActionWorkspaceParams()
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    trifinger_lcm = TrifingerLCMService()
    print("Resetting Trifinger Position...")
    trifinger_lcm.execute_trajectory(action_params.get_reset_knot(), no_data=True)
    new_trajectory = None
    epochs = 0

    # Create learnable system
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    learned_model = LearnedModel()
    learned_model.write_to_file("init")
    configure_solvers(nvar=len(mjx.make_data(learned_model.base_model).efc_J))
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    learned_traj = LearnedTrajectory(base_model=learned_model.active_model)
    learned_traj.write_to_file("init")

    # Sample initial action (from true obj pose)
    # action_cem = ActionCEM(action_params)
    selected_action = action_params.random_action()
    selected_knots = np.stack(
        [action_params.get_reset_knot(), action_params.get_reset_knot()]
    )
    start_true_object_pose = (
        trifinger_lcm.get_current_object_pose() if start_with_true_object_pose else None
    )
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
    print("JIT Mujoco XLA Step...")
    gui_vis = MJXMeshcatVisualizer(
        learned_model.active_model,
        mjx_util.jit_forward(
            learned_model.active_model, mjx.make_data(learned_model.active_model)
        ),
    )
    gui_vis.update_visuals(
        learned_model.active_model,
        [
            mjx_util.jit_forward(
                learned_model.active_model, mjx.make_data(learned_model.active_model)
            )
        ],
        (
            {
                trifinger_lcm.object_geom_name: [
                    (
                        start_true_object_pose[4:],
                        Rotation.from_quat(
                            start_true_object_pose[:4], scalar_first=True
                        ),
                    )
                ]
            }
            if start_true_object_pose is not None
            else None
        ),
    )
    gui_vis.draw_action_samples(selected_knots[np.newaxis, :, :])

    # Create trajectory set
    dataset = TrajectorySet()

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "a - [WIP] Action selection\n"
            "e - Execute selected action + collect data\n"
            "l - Load trajectory data\n"
            "t - Train on collected data\n"
            "b - breakpoint()\n"
            "h - Print Help\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        if signal_pressed:
            print("Ctrl-C pressed and not cleared, exiting!")
            break

        command_char = input("Command $ ")
        if len(command_char) > 1:
            print("Please only command one character.")
            continue

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

        elif command_char == "a":
            # test_obs = observed_info(
            #     (learned_model.params, learned_traj.get_full_trajectory()),
            #     dataset.full_trajectory(),
            #     learned_model.base_model,
            #     InfoHyperparameters(),
            # )
            test_exp = expected_info(
                dataset.full_trajectory()["ctrl"],
                (learned_model.params, learned_traj.final_q),
                [
                    geom_name
                    for geom_name in dataset.full_trajectory().keys()
                    if isinstance(dataset.full_trajectory()[geom_name], dict)
                    and "contact_normal_W" in dataset.full_trajectory()[geom_name]
                ],
                learned_model.base_model,
                InfoHyperparameters(),
            )
            breakpoint()

        elif command_char == "e" or command_char == "l":
            if command_char == "e":
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
                ctrlqs = []
                ctrlvs = []
                for geom_idx, geom_name in enumerate(
                    trifinger_lcm.fingertip_geom_names
                ):
                    ctrlqs.append(ctrl_total[:, geom_idx * 3 : (geom_idx + 1) * 3])
                    ctrlvs.append(
                        ctrl_total[:, (n_q + geom_idx * 3) : (n_q + (geom_idx + 1) * 3)]
                    )
                    new_trajectory[geom_name]["ctrl"] = np.concatenate(
                        [ctrlqs[-1], ctrlvs[-1]],
                        axis=-1,
                    )
                # TODO: assumes trifinger order == mjcf actuator order!
                new_trajectory["ctrl"] = jnp.concatenate(ctrlqs + ctrlvs, axis=-1)

                # Write data to TrajectorySet
                dataset.add_trajectory(new_trajectory)

                # Expand LearnedTrajectorynew_tratrajjectory
                learned_traj.extend_traj(len(new_trajectory["time"]))

            elif command_char == "l":
                try:
                    load_file = str(input("Path to trajectory data? "))
                    # TODO: Move to file_utils
                    with open(load_file, "rb") as file:
                        new_trajs = pickle.load(file)
                    for traj in new_trajs:
                        dataset.add_trajectory(traj)
                        learned_traj.extend_traj(len(traj["time"]))
                except Exception as exc:
                    print(f"Error loading data: {exc}")
                    continue

            # Visualize Complete Data
            gui_vis.update_visuals(
                learned_model.active_model,
                gui_vis.data_trajectory
                + [gui_vis.data_trajectory[-1]]
                * (
                    len(dataset.full_trajectory()["time"])
                    - len(gui_vis.data_trajectory)
                ),
                {
                    trifinger_lcm.object_geom_name: [
                        (
                            row[4:],
                            Rotation.from_quat(row[:4], scalar_first=True),
                        )
                        for row in dataset.full_trajectory()[
                            trifinger_lcm.object_geom_name
                        ]["position"]
                    ],
                }
                | {
                    geom_name: [
                        (
                            row,
                            Rotation.identity(),
                        )
                        for row in dataset.full_trajectory()[geom_name]["position"]
                    ]
                    for geom_name in trifinger_lcm.fingertip_geom_names
                },
            )

            ## END Collect New Data

        elif command_char == "t":
            ## Train on data
            if len(dataset) == 0:
                print("Cannot train without data.\n")
                continue

            try:
                n_epochs = int(input("How many epochs? "))
            except ValueError:
                print("Cancelling...")
                continue
            print("Training...")
            train_epochs(
                learned_model,
                learned_traj,
                dataset,
                n_epochs,
                epochs,
                gui_vis,
            )
            epochs = epochs + n_epochs

            ## END Train on data

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
    # TODO: move gin external config to separate module
    gin.register(np.array, module="np")
    gin.register(np.random.uniform, module="np.random")
    gin.register(optax.adam, module="optax")
    print(f"Loading Config File: {get_config(args.config_file)}")
    gin.parse_config_file(get_config(args.config_file))
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    main(config_file=args.config_file)


if __name__ == "__main__":
    main_fn()
