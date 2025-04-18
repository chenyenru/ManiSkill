import numpy as np
import argparse
from pathlib import Path
import h5py
from mani_skill.utils.logging_utils import logger

from mani_skill.utils.io_utils import dump_json, load_json


def merge_trajectories(output_path: str, traj_paths: list, recompute_id: bool = True):
    """
    Merges multiple JSON and H5 files into a single JSON and H5 file.

    This function combines the contents of multiple JSON and H5 files. It keeps the first value for all keys
    (other than "episodes") and logs a warning for any differences. The "episodes" from each JSON file are merged
    into a single list, and the corresponding H5 data is copied to the output H5 file.

    Args:
        output_path (str): The path to the output H5 file. The corresponding JSON file will be saved with the same
                           name but with a .json extension.
        traj_paths (list): A list of paths to the input trajectory files (H5 files). The corresponding JSON files
                           should have the same name but with a .json extension.
        recompute_id (bool): If True, recompute the episode IDs to ensure they are unique. If False, keep the original
                             episode IDs.

    Raises:
        AssertionError: If there is a conflict in the episode IDs when recompute_id is False.
    """
    logger.info(f"Merging {output_path}")

    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"episodes": []}
    cnt = 0

    for traj_path in traj_paths:
        traj_path = str(traj_path)
        logger.info(f"Merging{traj_path}")

        with h5py.File(traj_path, "r") as h5_file:
            json_data = load_json(traj_path.replace(".h5", ".json"))
            
            # For keys other than episodes, keep the first data
            # and check if there is any conflict with other data.
            for key, value in json_data.items():
                if key == "episodes":
                    continue
                if key not in merged_json_data:
                    merged_json_data[key] = value
                else:
                    if merged_json_data[key] != value:
                        logger.warning(f"Conflict detected for key {key} in {traj_path}: {merged_json_data[key]} != {value}")

            # Merge episodes
            for ep in json_data["episodes"]:
                episode_id = ep["episode_id"]
                traj_id = f"traj_{episode_id}"

                # Copy h5 data
                if recompute_id:
                    new_traj_id = f"traj_{cnt}"
                else:
                    new_traj_id = traj_id

                assert new_traj_id not in merged_h5_file, new_traj_id
                h5_file.copy(traj_id, merged_h5_file, new_traj_id)

                # Copy json data
                if recompute_id:
                    ep["episode_id"] = cnt
                merged_json_data["episodes"].append(ep)

                cnt += 1

    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)

def merge_trajectory_percentage(output_path: str, traj_paths: list, behavior_distribution: list, recompute_id: bool = True, total_num_demos: int = 100):
    """
    Merges multiple JSON and H5 files into a single JSON and H5 file based on behavior distribution.

    Args:
        output_path (str): The path to the output H5 file. The corresponding JSON file will be saved with the same
                           name but with a .json extension.
        traj_paths (list): A list of paths to the input trajectory files (H5 files). The corresponding JSON files
                           should have the same name but with a .json extension.
        behavior_distribution (list): A list of percentages indicating the distribution of behaviors.
        recompute_id (bool): If True, recompute the episode IDs to ensure they are unique. If False, keep the original
                             episode IDs.

    Raises:
        AssertionError: If there is a conflict in the episode IDs when recompute_id is False.
    """
    # print(len(traj_paths), len(behavior_distribution))
    assert len(traj_paths) == len(behavior_distribution), "The number of trajectory paths must match the number of behavior distributions."
    assert sum(behavior_distribution) == 100, "The sum of behavior distributions must be 100%."

    logger.info(f"Merging {output_path} with behavior distribution {behavior_distribution}")

    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"episodes": []}
    cnt = 0

    for traj_path, percentage in zip(traj_paths, behavior_distribution):
        traj_path = str(traj_path)
        logger.info(f"Merging {traj_path} with {percentage}% of episodes")

        with h5py.File(traj_path, "r") as h5_file:
            json_data = load_json(traj_path.replace(".h5", ".json"))
            num_episodes = len(json_data["episodes"])
            assert num_episodes >= total_num_demos, f"Number of episodes {num_episodes} is less than total number of demos {total_num_demos}."
            num_selected_episodes = int(total_num_demos * (percentage / 100))

            # For keys other than episodes, keep the first data
            # and check if there is any conflict with other data.
            for key, value in json_data.items():
                if key == "episodes":
                    continue
                if key not in merged_json_data:
                    merged_json_data[key] = value
                else:
                    if merged_json_data[key] != value:
                        logger.warning(f"Conflict detected for key {key} in {traj_path}: {merged_json_data[key]} != {value}")

            # Merge selected episodes
            selected_episodes = np.random.choice(json_data["episodes"], num_selected_episodes, replace=False)
            for ep in selected_episodes:
                episode_id = ep["episode_id"]
                traj_id = f"traj_{episode_id}"

                # Copy h5 data
                if recompute_id:
                    new_traj_id = f"traj_{cnt}"
                else:
                    new_traj_id = traj_id

                assert new_traj_id not in merged_h5_file, new_traj_id
                h5_file.copy(traj_id, merged_h5_file, new_traj_id)

                # Copy json data
                if recompute_id:
                    ep["episode_id"] = cnt
                merged_json_data["episodes"].append(ep)

                cnt += 1

    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", nargs="+")
    parser.add_argument("-o", "--output-path", type=str)
    parser.add_argument("-p", "--pattern", type=str, default="trajectory.h5")
    parser.add_argument("--behavior-distribution", type=float, nargs="+", default=None)
    parser.add_argument("--traj-paths", type=str, nargs="+", default=None)
    parser.add_argument("-n", "--total-num-demos", type=int, default=100)
    args = parser.parse_args()

    traj_paths = []
    if args.input_dirs is not None:
        for input_dir in args.input_dirs:
            input_dir = Path(input_dir)
            traj_paths.extend(sorted(input_dir.rglob(args.pattern)))

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.behavior_distribution and args.traj_paths:
        merge_trajectory_percentage(args.output_path, args.traj_paths, args.behavior_distribution, total_num_demos=args.total_num_demos)
    else:
        merge_trajectories(args.output_path, traj_paths)


if __name__ == "__main__":
    main()
