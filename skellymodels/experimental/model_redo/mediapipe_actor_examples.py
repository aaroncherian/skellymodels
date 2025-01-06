from pathlib import Path
import numpy as np

from skellymodels.experimental.model_redo.managers.actor import Actor
from skellymodels.experimental.model_redo.utils.create_mediapipe_actor import create_aspects_for_mediapipe_human, split_data
from pprint import pprint

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


# f = 2