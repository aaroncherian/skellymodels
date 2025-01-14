from pathlib import Path
import numpy as np
import pandas as pd

from skellymodels.experimental.model_redo.managers.human import Human, HumanConfiguration, HumanAspects

from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.calculate_center_of_mass import calculate_center_of_mass_from_trajectory
from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.enforce_rigid_bones import enforce_rigid_bones_from_trajectory
from pprint import pprint

## Choose a path to the directory 
path_to_data = Path(r"C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_sample_data\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")
data = np.load(path_to_data)

## Create an Actor
human = Human(name="human_one", 
              configuration=HumanConfiguration(tracker_type='mediapipe', 
                                               include_face=True, 
                                               include_hands=True))

human.from_tracked_points_numpy(tracked_points_numpy_array=data)
pprint([human.aspects.values()])

# Calculate center of mass 
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
f = 2