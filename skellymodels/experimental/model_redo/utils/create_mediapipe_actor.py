from skellymodels.experimental.model_redo.managers.actor import Actor
from skellymodels.experimental.model_redo.models import Aspect
from skellymodels.experimental.model_redo.builders.anatomical_structure_builder import AnatomicalStructureBuilder

from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import MediapipeModelInfo

import numpy as np



def create_aspects_for_mediapipe_human():
    body = Aspect(name = "body")
    body_structure = (AnatomicalStructureBuilder()
             .with_landmarks(MediapipeModelInfo().body_landmark_names)
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
             .with_landmarks(face_landmark_names)
             .build()
    )
    face.add_anatomical_structure(face_structure)
    face.add_tracker_type("mediapipe")

    left_hand = Aspect(name = "left_hand")
    left_hand_landmark_names = [ f"left_{str(i).zfill(4)}" for i in range(MediapipeModelInfo().num_tracked_points_left_hand)]
    left_hand_structure = (AnatomicalStructureBuilder()
                .with_landmarks(left_hand_landmark_names)
                .build()
        )
    left_hand.add_anatomical_structure(left_hand_structure)
    left_hand.add_tracker_type("mediapipe")

    right_hand = Aspect(name = "right_hand")
    right_hand_landmark_names = [f"right_{str(i).zfill(4)}" for i in range(MediapipeModelInfo().num_tracked_points_right_hand)]
    right_hand_structure = (AnatomicalStructureBuilder()
                .with_landmarks(right_hand_landmark_names)
                .build()
        )
    right_hand.add_anatomical_structure(right_hand_structure)
    right_hand.add_tracker_type("mediapipe")

    return body, right_hand, left_hand, face


def split_data(data: np.ndarray) -> dict:
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
    category_data = {name: data[:,slc,:] for name, slc in slices.items()}
    
    return category_data
