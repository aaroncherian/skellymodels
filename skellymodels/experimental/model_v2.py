from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from skellymodels.experimental.validators import (LandmarkValidator,
                                                  VirtualMarkerValidator,
                                                  SegmentConnectionsValidator,
                                                  CenterOfMassValidator)
from skellymodels.model_info.qualisys_model_info import QualisysModelInfo
# from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
import numpy as np
from skellymodels.experimental.fmc_files.calculate_center_of_mass import calculate_center_of_mass_from_aspect

@dataclass
class AnatomicalStructure:
    landmark_names: List[str]
    virtual_markers_definitions: Optional[Dict[str, Dict[str, List[Union[float, str]]]]] = None
    segment_connections: Optional[Dict[str, Dict[str, str]]] = None
    center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def marker_names(self):
        markers = self.landmark_names.copy()
        if self.virtual_markers_definitions:
            markers.extend(self.virtual_markers_definitions.keys())
        return markers

    @property
    def virtual_marker_names(self):
        if not self.virtual_markers_definitions:
            return []
        return list(self.virtual_markers_definitions.keys())



from pydantic import BaseModel, model_validator, ConfigDict

class TrajectoryValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data:np.ndarray
    marker_names: List[str]

    @model_validator(mode="after")
    def validate_data(self):
        if self.data.shape[1] != len(self.marker_names):
            raise ValueError(f"Trajectory data must have the same number of markers as input name list. Data has {self.data.shape[1]} markers and list has {len(self.marker_names)} markers.")

class Trajectory:
    def __init__(self, name: str, data: np.ndarray, marker_names: List[str], virtual_marker_definitions: Dict = None):
        self.name = name
        self._trajectories = {}
        self._marker_names = marker_names
        self._virtual_marker_definitions = virtual_marker_definitions
        self._validate_data(data=data, marker_names=marker_names)
        self._set_trajectory_data(data=data, marker_names=marker_names, virtual_marker_definitions=virtual_marker_definitions)


    def _validate_data(self, data: np.ndarray, marker_names: List[str]):
        TrajectoryValidator(data=data, marker_names= marker_names)

    def _set_trajectory_data(self, data:np.ndarray, marker_names:List[str], virtual_marker_definitions: Dict = None):
        self._trajectories.update({marker_name: data[:, i, :] for i, marker_name in enumerate(marker_names)})

        if virtual_marker_definitions:
            print(f'Calculating virtual markers: {list(virtual_marker_definitions.keys())}')
            virtual_marker_data = {}
            for vm_name, vm_info in virtual_marker_definitions.items():
                vm_positions = np.zeros((data.shape[0], 3))
                for marker_name, weight in zip(vm_info["marker_names"], vm_info["marker_weights"]):
                    vm_positions += self._trajectories[marker_name] * weight
                virtual_marker_data[vm_name] = vm_positions

            self._trajectories.update(virtual_marker_data)

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def landmark_trajectories(self):
        return {marker_name:trajectory for marker_name, trajectory in self._trajectories.items() if marker_name in self._marker_names}

    @property
    def virtual_marker_trajectories(self):
        if not self._virtual_marker_definitions:
            return {}
        return {marker_name:trajectory for marker_name, trajectory in self._trajectories.items() if marker_name in self._virtual_marker_definitions.keys()}

    def get_marker(self, marker_name: str):
        return self._trajectories[marker_name]

    def get_frame(self, frame_number: int):
        return {marker_name: trajectory[frame_number] for marker_name, trajectory in self._trajectories.items()}




class AnatomicalStructureBuilder:
    def __init__(self):
        self.landmark_names: Optional[List[str]] = None
        self.virtual_markers_definitions: Optional[Dict[str, Dict[str, List[Union[float, str]]]]] = None
        self.segment_connections: Optional[Dict[str, Dict[str, str]]] = None
        self.center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def _marker_names(self):
        if not self.landmark_names:
            raise ValueError("Landmark names must be set before calling for a marker list.")
        markers = self.landmark_names.copy()
        if self.virtual_markers_definitions:
            markers.extend(self.virtual_markers_definitions.keys())
        return markers

    def with_landmarks(self, landmark_names: List[str]):
        LandmarkValidator(landmark_names=landmark_names)
        self.landmark_names = landmark_names.copy()
        return self

    def with_virtual_markers(self, virtual_marker_definitions: Dict[str, Dict[str, List[Union[float, str]]]]):
        if not self.landmark_names:
            raise ValueError("Landmark names must be set before adding virtual markers.")
        VirtualMarkerValidator(virtual_markers=virtual_marker_definitions,
                               landmark_names=self.landmark_names)
        self.virtual_markers_definitions = virtual_marker_definitions
        return self

    def with_segment_connections(self, segment_connections: Dict[str, Dict[str, str]]):
        SegmentConnectionsValidator(segment_connections=segment_connections,
                                    marker_names=self._marker_names)
        self.segment_connections = segment_connections
        return self

    def with_center_of_mass(self, center_of_mass_definitions: Dict[str, Dict[str, float]]):
        if not self.segment_connections:
            raise ValueError("Segment connections must be set before adding center of mass definitions")
        CenterOfMassValidator(center_of_mass_definitions=center_of_mass_definitions,
                                segment_connections=self.segment_connections)
        self.center_of_mass_definitions = center_of_mass_definitions
        return self

    def build(self):
        if not self.landmark_names:
            raise ValueError("Cannot build AnatomicalStructure without landmark names")
        return AnatomicalStructure(landmark_names=self.landmark_names,
                                   virtual_markers_definitions=self.virtual_markers_definitions,
                                   segment_connections=self.segment_connections,
                                   center_of_mass_definitions=self.center_of_mass_definitions)



class Aspect:
    def __init__(self, name:str):
        self.name = name
        self.anatomical_structure = {}
        self.trajectories = {}
        self.metadata = {}

    def add_anatomical_structure(self, anatomical_structure: AnatomicalStructure):
        self.anatomical_structure = anatomical_structure

    def add_trajectories(self, name:str, trajectory:np.ndarray):
        """Add a complete set of trajectories including all markers"""
        self.trajectories[name] = Trajectory(name=name,
                                       data=trajectory,
                                       marker_names = self.anatomical_structure.marker_names,
                                       virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions)

    def add_landmark_trajectories(self, trajectory: np.ndarray):
        """Add trajectories for basic landmarks, calculating virtual markers if defined"""
        self.trajectories['main'] = Trajectory(name="main",
                                       data=trajectory,
                                       marker_names = self.anatomical_structure.landmark_names,
                                       virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions)


class Actor:
    def __init__(self, name: str):
        self.name = name
        self.aspects = {}

    def __getitem__(self, key: str):
        return self.aspects[key]

    def __str__(self):
        return str(self.aspects.keys())

    def add_aspect(self, aspect: Aspect):
        self.aspects[aspect.name] = aspect

    def get_data(self, aspect_name:str, type:str):
        return self.aspects[aspect_name].trajectories[type].trajectories

    def get_marker_data(self, aspect_name:str, type:str, marker_name:str):
        return self.aspects[aspect_name].trajectories[type].get_marker(marker_name)

    def get_frame(self, aspect_name:str, type:str, frame_number:int):
        return self.aspects[aspect_name].trajectories[type].get_frame(frame_number)




class Human(Actor):
    pass








from pathlib import Path
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import MediapipeModelInfo


def build_human_from_mediapipe_model_info(human:Actor, data:np.ndarray):

    body = Aspect(name = "body")
    body_structure = (AnatomicalStructureBuilder()
             .with_landmarks(MediapipeModelInfo().body_landmark_names)
             .with_virtual_markers(MediapipeModelInfo().virtual_markers_definitions)
             .with_segment_connections(MediapipeModelInfo().segment_connections)
             .with_center_of_mass(MediapipeModelInfo().center_of_mass_definitions)
             .build()
    )
    body.add_anatomical_structure(body_structure)

    face = Aspect(name = "face")
    face_landmark_names = [str(i).zfill(4) for i in range(MediapipeModelInfo().num_tracked_points_face)]
    face_structure = (AnatomicalStructureBuilder()
             .with_landmarks(face_landmark_names)
             .build()
    )
    face.add_anatomical_structure(face_structure)

    left_hand = Aspect(name = "left_hand")
    left_hand_landmark_names = [ str(i).zfill(4) for i in range(MediapipeModelInfo().num_tracked_points_left_hand)]
    left_hand_structure = (AnatomicalStructureBuilder()
                .with_landmarks(left_hand_landmark_names)
                .build()
        )
    left_hand.add_anatomical_structure(left_hand_structure)

    right_hand = Aspect(name = "right_hand")
    right_hand_landmark_names = [ str(i).zfill(4) for i in range(MediapipeModelInfo().num_tracked_points_right_hand)]
    right_hand_structure = (AnatomicalStructureBuilder()
                .with_landmarks(right_hand_landmark_names)
                .build()
        )
    right_hand.add_anatomical_structure(right_hand_structure)

    data_split_by_category = split_data(data)

    body.add_landmark_trajectories(data_split_by_category['pose_landmarks'])
    face.add_landmark_trajectories(data_split_by_category['face_landmarks'])
    left_hand.add_landmark_trajectories(data_split_by_category['left_hand_landmarks'])
    right_hand.add_landmark_trajectories(data_split_by_category['right_hand_landmarks'])

    human.add_aspect(body)
    human.add_aspect(face)
    human.add_aspect(left_hand)
    human.add_aspect(right_hand)

    return human

    f = 2


import numpy as np

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


path_to_data = Path(r"C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_sample_data\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")

data = np.load(path_to_data)


human = build_human_from_mediapipe_model_info(Actor(name="human_one"), data)

for aspect in human.aspects.values():
    calculate_center_of_mass_from_aspect()

f = 2


# body_structure = (AnatomicalStructureBuilder()
#              .with_landmarks(MediapipeModelInfo().landmark_names)
#              .with_virtual_markers(MediapipeModelInfo().virtual_markers_definitions)
#              .with_segment_connections(MediapipeModelInfo().segment_connections)
#              .with_center_of_mass(MediapipeModelInfo().center_of_mass_definitions)
#              .build()
# )

# body = Aspect(name="body")
# body.add_anatomical_structure(body_structure)

# # data = np.load(path_to_data)
# body.add_landmark_trajectories(data)

# human.add_aspect(body)
# human.get_data(aspect_name = 'body', type='main')
# f = 2
