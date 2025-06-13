from copy import deepcopy
import numpy as np
from typing import Dict, List, Union,Tuple
from collections import deque


# def calculate_bone_lengths_and_statistics(
#     segment_data: Dict[str, Dict[str, np.ndarray]]
# ) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
#     """
#     Calculates bone lengths for each frame and their statistics (median and standard deviation)
#     based on marker positions and segment connections.

#     Parameters:
#     - segment_data: A dictionary defining the segments (bones) with segment names as keys
#       and dictionaries with 'proximal' and 'distal' marker xyz locations as values.

#     Returns:
#     - A dictionary with segment names as keys and dictionaries with lengths, median lengths,
#       and standard deviations as values.
#     """
#     bone_statistics = {}

#     for segment_name, segment in segment_data.items():
#         proximal_positions = segment['proximal']
#         distal_positions = segment['distal']

#         lengths = np.linalg.norm(distal_positions - proximal_positions, axis=1)

#         median_length = np.nanmedian(lengths)
#         stdev_length = np.nanstd(lengths)

#         bone_statistics[segment_name] = {
#             "lengths": lengths, 
#             "median": median_length, 
#             "stdev": stdev_length
#         }

#     return bone_statistics

def calculate_bone_lengths_and_statistics(
        marker_trajectories: Dict[str, np.ndarray],
        joint_hierarchy: Dict[str, List[str]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    bone_stats = {}
    for parent, children in joint_hierarchy.items():
        parent_xyz = marker_trajectories[parent]
        print(f"\nBone lengths from '{parent}' to:")

        for child in children:
            child_xyz = marker_trajectories[child]
            lengths = np.linalg.norm(child_xyz - parent_xyz, axis=1)

            median = np.nanmedian(lengths)
            stdev  = np.nanstd(lengths)

            bone_stats[(parent, child)] = {
                "median": median,
                "stdev":  stdev,
            }

            print(f"  - '{child}': median = {median:.2f}, stdev = {stdev:.2f}")
    return bone_stats

def rigidify_bones(
    marker_data: Dict[str, np.ndarray],
    segment_connections: Dict[str, Dict[str, np.ndarray]],
    bone_lengths_and_statistics: Dict[str, Dict[str, Union[np.ndarray, float]]],
    joint_hierarchy: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Enforces rigid bones by adjusting the distal joints of each segment to match the median length.

    Parameters:
    - marker_data: The original marker positions.
    - segment_connections: Information about how segments (bones) are connected.
    - bone_lengths_and_statistics: The desired bone lengths and statistics for each segment.
    - joint_hierarchy: The hierarchy of joints, indicating parent-child relationships.

    Returns:
    - A dictionary of adjusted marker positions.
    """
    rigid_marker_data = deepcopy(marker_data)

    for segment_name, stats in bone_lengths_and_statistics.items():
        desired_length = stats["median"]
        lengths = stats["lengths"]

        segment = segment_connections[segment_name]
        proximal_marker, distal_marker = segment['proximal'], segment['distal']

        for frame_index, current_length in enumerate(lengths):
            if np.isnan(current_length) or current_length < 1e-10:
                continue
            if abs(current_length - desired_length) > 1e-6:
                proximal_position = rigid_marker_data[proximal_marker][frame_index]
                distal_position = rigid_marker_data[distal_marker][frame_index]
                direction = distal_position - proximal_position
                try:
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:  # If effectively zero
                        direction = np.array([0.0, 0.0, 1.0])  # Use a default direction
                    else:
                        direction = direction / norm  # Proper normalization
                except:
                    direction = np.array([0.0, 0.0, 1.0])  # Fallback to default
                adjustment = (desired_length - current_length) * direction

                rigid_marker_data[distal_marker][frame_index] += adjustment

                adjust_children(distal_marker, frame_index, adjustment, rigid_marker_data, joint_hierarchy)

    return rigid_marker_data


def adjust_children(
    parent_marker: str,
    frame_index: int,
    adjustment: np.ndarray,
    marker_data: Dict[str, np.ndarray],
    joint_hierarchy: Dict[str, List[str]],
):
    """
    Recursively adjusts the positions of child markers based on the adjustment of the parent marker.
    """
    if parent_marker in joint_hierarchy:
        for child_marker in joint_hierarchy[parent_marker]:
            marker_data[child_marker][frame_index] += adjustment
            adjust_children(child_marker, frame_index, adjustment, marker_data, joint_hierarchy)


def merge_rigid_marker_data(rigid_marker_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Merges the center of mass data from multiple segments into a single array.

    Parameters:
    - segment_com_data: A dictionary where each key is a segment name and the value is the center of mass data for that segment.

    Returns:
    - A numpy array containing the merged center of mass data.
    """

    rigid_marker_data_list = list(rigid_marker_data.values())

    return np.stack(rigid_marker_data_list, axis=1)



def rigidify_forward_pass(
        marker_trajectories: Dict[str, np.ndarray],
        joint_hierarchy: Dict[str, List[str]],
        bone_stats: Dict[str, Dict[str, Union[np.ndarray, float]]],
):
    marker_names = list(marker_trajectories.keys())
    data = np.stack([marker_trajectories[n] for n in marker_names], axis=1)
    num_frames, num_markers, _ = data.shape

    root = next(iter(joint_hierarchy)) 
    edge_connections, queue = [], deque([root])
    while queue:
        parent = queue.popleft()
        for child in joint_hierarchy.get(parent, []):
            edge_connections.append((parent, child))
            queue.append(child)

    rigid_data = data.copy()
    for frame in range(num_frames):
        frame_data = rigid_data[frame]
        direction_vector = np.array([1.0,0.0,0.0])

        for parent, child in edge_connections:
            parent_index, child_index = marker_names.index(parent), marker_names.index(child)
            vector = frame_data[child_index] - frame_data[parent_index]
            norm = np.linalg.norm(vector)
            if np.isfinite(norm) and norm > 1e-6:
                direction_vector = vector/norm
            rigid_length = bone_stats[(parent,child)]["median"]
            frame_data[child_index] = frame_data[parent_index] + direction_vector*rigid_length

    return {name: rigid_data[:, i, :] for i, name in enumerate(marker_names)}




    

def enforce_rigid_bones(marker_trajectories:np.ndarray,
                        joint_hierarchy: Dict[str, List[str]]
                        ):
    
    bone_lengths_and_statistics = calculate_bone_lengths_and_statistics(
        marker_trajectories=marker_trajectories,
        joint_hierarchy=joint_hierarchy
    )

    rigid_marker_data = rigidify_forward_pass(
        marker_trajectories=marker_trajectories,
        bone_stats=bone_lengths_and_statistics,
        joint_hierarchy=joint_hierarchy
    )
    
    f = 2
    return merge_rigid_marker_data(rigid_marker_data)