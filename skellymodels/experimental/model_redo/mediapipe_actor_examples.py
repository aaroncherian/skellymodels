from pathlib import Path
import numpy as np
import pandas as pd

from skellymodels.experimental.model_redo.managers.actor import Actor
from skellymodels.experimental.model_redo.utils.create_mediapipe_actor import create_aspects_for_mediapipe_human, split_data
from skellymodels.experimental.model_redo.models import Aspect
from skellymodels.experimental.model_redo.builders.anatomical_structure_builder import AnatomicalStructureBuilder

from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import MediapipeModelInfo
from pprint import pprint

from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.calculate_center_of_mass import calculate_center_of_mass_from_trajectory
from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.enforce_rigid_bones import enforce_rigid_bones_from_trajectory

### CHOOSE DIRECTORY PATH
path_to_data = Path(r"C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_sample_data\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")
data = np.load(path_to_data)

### CREATE AN ACTOR
human = Actor(name="human_one")

### CREATING ASPECTS
body = Aspect(name = "body")
body_structure = (AnatomicalStructureBuilder()
            .with_tracked_points(MediapipeModelInfo().body_landmark_names)
            .with_virtual_markers(MediapipeModelInfo().virtual_markers_definitions)
            .with_segment_connections(MediapipeModelInfo().segment_connections)
            .with_center_of_mass(MediapipeModelInfo().center_of_mass_definitions)
            .with_joint_hierarchy(MediapipeModelInfo().joint_hierarchy)
            .build()
)
body.add_anatomical_structure(body_structure)
body.add_tracker_type("mediapipe")

face = Aspect(name = "face")
face_landmark_names = [str(i).zfill(4) for i in range(MediapipeModelInfo().num_tracked_points_face)]
face_structure = (AnatomicalStructureBuilder()
            .with_tracked_points(face_landmark_names)
            .build()
)
face.add_anatomical_structure(face_structure)
face.add_tracker_type("mediapipe")

left_hand = Aspect(name = "left_hand")
left_hand_landmark_names = [ f"left_{str(i).zfill(4)}" for i in range(MediapipeModelInfo().num_tracked_points_left_hand)]
left_hand_structure = (AnatomicalStructureBuilder()
            .with_tracked_points(left_hand_landmark_names)
            .build()
    )
left_hand.add_anatomical_structure(left_hand_structure)
left_hand.add_tracker_type("mediapipe")

right_hand = Aspect(name = "right_hand")
right_hand_landmark_names = [f"right_{str(i).zfill(4)}" for i in range(MediapipeModelInfo().num_tracked_points_right_hand)]
right_hand_structure = (AnatomicalStructureBuilder()
            .with_tracked_points(right_hand_landmark_names)
            .build()
    )
right_hand.add_anatomical_structure(right_hand_structure)
right_hand.add_tracker_type("mediapipe")


### SPLIT 3D DATA 
# split data into body/hands/face based off of old model info
tracked_object_names = MediapipeModelInfo.tracked_object_names
face_landmark_names = [str(i).zfill(4) for i in range(MediapipeModelInfo().num_tracked_points_face)]

lengths = [
    len(MediapipeModelInfo.body_landmark_names), 
    len(MediapipeModelInfo.hand_landmark_names), 
    len(MediapipeModelInfo.hand_landmark_names),
    len(face_landmark_names)         
]

# Generate slices for each category
current_index = 0
slices = {}
for name, length in zip(tracked_object_names, lengths):
    slices[name] = slice(current_index, current_index + length)
    current_index += length

# Split the data using slices
data_split_by_category = {name: data[:,slc,:] for name, slc in slices.items()}

body.add_tracked_points(data_split_by_category['pose_landmarks'])
face.add_tracked_points(data_split_by_category['face_landmarks'])
left_hand.add_tracked_points(data_split_by_category['left_hand_landmarks'])
right_hand.add_tracked_points(data_split_by_category['right_hand_landmarks'])

## Add the aspects to the actor
human.add_aspect(body)
human.add_aspect(face)
human.add_aspect(left_hand)
human.add_aspect(right_hand)
pprint([human.aspects.values()])

### ANATOMICAL PIPELINE FUNCTIONS
for aspect in human.aspects.values():
    if aspect.anatomical_structure.center_of_mass_definitions:
        print('Calculating center of mass for aspect:', aspect.name)
        total_body_com, segment_com = calculate_center_of_mass_from_trajectory(aspect.trajectories['3d_xyz'], aspect.anatomical_structure.center_of_mass_definitions)

        aspect.add_trajectory(name = 'total_body_com',
                                data = total_body_com,
                                marker_names = ['total_body'])
        
        aspect.add_trajectory(name = 'segment_com',
                                data = segment_com,
                                marker_names = list(aspect.anatomical_structure.center_of_mass_definitions.keys()))
        
    else:
        print('Skipping center of mass calculation for aspect:', aspect.name)
pprint([human.aspects.values()])

for aspect in human.aspects.values():
    if aspect.anatomical_structure.joint_hierarchy:
        print('Enforcing rigid bones for aspect:', aspect.name)
        rigid_bones = enforce_rigid_bones_from_trajectory(aspect.trajectories['3d_xyz'], aspect.anatomical_structure.joint_hierarchy)

        aspect.add_trajectory(name = 'rigid_3d_xyz',
                                data = rigid_bones,
                                marker_names = aspect.anatomical_structure.marker_names)
    else:
        print('Skipping rigid bones enforcement for aspect:', aspect.name)
pprint([human.aspects.values()])

### SAVING OUT DATA
# Uncomment the below if you want to run it, but the saving stuff is also written out in the iPython notebook

# def save_out_numpy_data(actor:Actor):
#     for aspect in actor.aspects.values():
#         for trajectory in aspect.trajectories.values():
#             print('Saving out numpy:', aspect.metadata['tracker_type'], aspect.name, trajectory.name)
#             np.save(f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.npy", trajectory.data)
            
# def save_out_csv_data(actor:Actor):
#     for aspect in actor.aspects.values():
#         for trajectory in aspect.trajectories.values():
#             print('Saving out CSV:', aspect.metadata['tracker_type'], aspect.name, trajectory.name)
#             trajectory.as_dataframe.to_csv(f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv")

# def save_out_all_data_csv(actor: Actor):
#     all_data = []

#     # Loop through aspects and trajectories
#     for aspect_name, aspect in actor.aspects.items():
#         for trajectory_name, trajectory in aspect.trajectories.items():
#             if trajectory_name == '3d_xyz':
#                 # Get tidy DataFrame for the trajectory
#                 trajectory_df = trajectory.as_dataframe
                
#                 # Add metadata column for model
#                 trajectory_df['model'] = f"{aspect.metadata['tracker_type']}_{aspect_name}"
                
#                 # Append DataFrame to the list
#                 all_data.append(trajectory_df)

#     # Combine all DataFrames into one
#     big_df = pd.concat(all_data, ignore_index=True)

#     # Sort by frame and then by model
#     big_df = big_df.sort_values(by=['frame', 'model']).reset_index(drop=True)

#     # Save the result to CSV
#     big_df.to_csv('freemocap_data_by_frame.csv', index=False)
#     print("Data successfully saved to 'freemocap_data_by_frame.csv'")



# save_out_numpy_data(actor = human)
# save_out_csv_data(actor = human)
# save_out_all_data_csv(actor = human)




f = 2

