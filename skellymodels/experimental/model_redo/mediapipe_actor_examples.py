from pathlib import Path
import numpy as np

from skellymodels.experimental.model_redo.managers.actor import Actor
from skellymodels.experimental.model_redo.utils.create_mediapipe_actor import create_aspects_for_mediapipe_human, split_data
from pprint import pprint

from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.calculate_center_of_mass import calculate_center_of_mass_from_trajectory
from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.enforce_rigid_bones import enforce_rigid_bones_from_trajectory
## Choose a path to the directory 
path_to_data = Path(r"C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_sample_data\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")
data = np.load(path_to_data)

## Create an Actor
human = Actor(name="human_one")

## Create body, hands and face aspects with anatomical structures using the mediapipe info model
body, right_hand, left_hand, face = create_aspects_for_mediapipe_human()
pprint(body)


## Split 3d data into body/face/hands and add to the respective aspects
data_split_by_category = split_data(data)

body.add_landmark_trajectories(data_split_by_category['pose_landmarks'])
face.add_landmark_trajectories(data_split_by_category['face_landmarks'])
left_hand.add_landmark_trajectories(data_split_by_category['left_hand_landmarks'])
right_hand.add_landmark_trajectories(data_split_by_category['right_hand_landmarks'])

## Add the aspects to the actor
human.add_aspect(body)
human.add_aspect(face)
human.add_aspect(left_hand)
human.add_aspect(right_hand)
pprint([human.aspects.values()])

# Calculate center of mass 
for aspect in human.aspects.values():
    if aspect.anatomical_structure.center_of_mass_definitions:
        print('Calculating center of mass for aspect:', aspect.name)
        total_body_com, segment_com = calculate_center_of_mass_from_trajectory(aspect.trajectories['3d_xyz'], aspect.anatomical_structure.center_of_mass_definitions)

        aspect.add_trajectories(name = 'total_body_com',
                                data = total_body_com,
                                marker_names = ['total_body'])
        
        aspect.add_trajectories(name = 'segment_com',
                                data = segment_com,
                                marker_names = list(aspect.anatomical_structure.center_of_mass_definitions.keys()))
        
    else:
        print('Skipping center of mass calculation for aspect:', aspect.name)
pprint([human.aspects.values()])

for aspect in human.aspects.values():
    if aspect.anatomical_structure.joint_hierarchy:
        print('Enforcing rigid bones for aspect:', aspect.name)
        rigid_bones = enforce_rigid_bones_from_trajectory(aspect.trajectories['3d_xyz'], aspect.anatomical_structure.joint_hierarchy)

        aspect.add_trajectories(name = 'rigid_3d_xyz',
                                data = rigid_bones,
                                marker_names = aspect.anatomical_structure.marker_names)
    else:
        print('Skipping rigid bones enforcement for aspect:', aspect.name)
pprint([human.aspects.values()])
f = 2


# f = 2