from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from skellymodels.experimental.validators import (LandmarkValidator, 
                                                  VirtualMarkerValidator, 
                                                  SegmentConnectionsValidator,
                                                  CenterOfMassValidator)
from skellymodels.model_info.qualisys_model_info import QualisysModelInfo
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
import numpy as np

    
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
    


from pydantic import BaseModel, field_validator, model_validator, ConfigDict

class TrajectoryValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data:np.ndarray
    marker_names: List[str]

    @model_validator(mode="after")
    def validate_data(self):
        if self.data.shape[1] != len(self.marker_names):
            raise ValueError(f"Trajectory data must have the same number of markers as anatomatical structure. Data has {self.data.shape[1]} markers and anatomical structure has {len(self.marker_names)} markers.")

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
        return {marker_name:trajectory for marker_name, trajectory in self._trajectories.items() if marker_name in self._virtual_marker_definitions.keys()}


    



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


class Character:
    def __init__(self, name: str):
        self.name = name
        self.aspects = {}

    def __getitem__(self, key: str):
        return self.aspects[key]
    
    def __str__(self):
        return self.name
    

class Aspect:
    def __init__(self, name:str):
        self.name = name
        self.anatomical_structure = {}
        self.trajectories = {}
        self.metadata = {}

    def add_anatomical_structure(self, anatomical_structure: AnatomicalStructure):
        self.anatomical_structure = anatomical_structure
    
    def add_landmark_trajectories(self, trajectory: np.ndarray):
        self.trajectories = Trajectory(name="landmarks", data=trajectory, marker_names = self.anatomical_structure.landmark_names, virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions)
        

skeleton = Character(name="mediapipe")

structure = (AnatomicalStructureBuilder()
             .with_landmarks(MediapipeModelInfo().landmark_names)
             .with_virtual_markers(MediapipeModelInfo().virtual_markers_definitions)
             .with_segment_connections(MediapipeModelInfo().segment_connections)
             .with_center_of_mass(MediapipeModelInfo().center_of_mass_definitions)
             .build()
)

aspect = Aspect(name="body")
aspect.add_anatomical_structure(structure)

from pathlib import Path
path_to_data = Path(r"C:\Users\Aaron\FreeMocap_Data\recording_sessions\freemocap_test_data\output_data\mediapipe_body_3d_xyz.npy")

data = np.load(path_to_data)

aspect.add_landmark_trajectories(data)


f = 2
