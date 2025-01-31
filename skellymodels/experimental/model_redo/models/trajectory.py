from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import Dict, List


class TrajectoryValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data:np.ndarray
    marker_names: List[str]

    @model_validator(mode="after")
    def validate_data(self):
        if self.data.shape[1] != len(self.marker_names):
            raise ValueError(f"Trajectory data must have the same number of markers as input name list. Data has {self.data.shape[1]} markers and list has {len(self.marker_names)} markers.")

    

class Trajectory:
    """
    A container for 3D motion capture trajectory data that handles both tracked markers 
    and derived virtual markers.

    The Trajectory class manages time-series 3D position data for a set of markers, supporting
    both directly tracked markers and calculated virtual markers. It provides methods to access
    the data in various formats (dictionary, numpy array, pandas DataFrame) and can compute
    virtual marker positions based on weighted combinations of tracked markers.

    Attributes:
        name (str): Identifier for this trajectory set
        _trajectories (Dict[str, np.ndarray]): Dictionary mapping marker names to their position data
            over time with shape (num_frames, 3)
        _tracked_point_names (List[str]): Names of the all tracked points (need to firm up some of these naming conventions)
         _marker_names (List[str]): Names of all markers (tracked + virtual) (repeated note about firming up naming conventions)
        _virtual_marker_definitions (Optional[Dict]): Definitions for calculating virtual markers
            from tracked markers using weighted combinations
        _segment_connections (Optional[Dict]): Definitions of segments connecting pairs of markers
        _num_frames (int): Total number of frames in the trajectory data

    """

    def __init__(self, name: str, 
                 data: np.ndarray, 
                 marker_names: List[str], 
                 virtual_marker_definitions: Dict | None = None, 
                 segment_connections: Dict | None = None):
        self.name = name
        self._trajectories = {}
        self._tracked_point_names = marker_names
        self._virtual_marker_definitions = virtual_marker_definitions
        self._segment_connections = segment_connections
        self._validate_data(data=data, marker_names=self._tracked_point_names)
        self._set_trajectory_data(data=data, marker_names=self._tracked_point_names, virtual_marker_definitions=virtual_marker_definitions)
        self._num_frames = data.shape[0]

    def _validate_data(self, data: np.ndarray, marker_names: List[str]):
        TrajectoryValidator(data=data, marker_names= marker_names)

    def _set_trajectory_data(self, data:np.ndarray, marker_names:List[str], virtual_marker_definitions: Dict = None):
        self._trajectories.update({marker_name: data[:, i, :] for i, marker_name in enumerate(marker_names)})
        self._marker_names = marker_names.copy()
        
        if virtual_marker_definitions:
            print(f'Calculating virtual markers: {list(virtual_marker_definitions.keys())}')
            virtual_marker_data = {}
            for vm_name, vm_info in virtual_marker_definitions.items():
                vm_positions = np.zeros((data.shape[0], 3))
                for marker_name, weight in zip(vm_info["marker_names"], vm_info["marker_weights"]):
                    vm_positions += self._trajectories[marker_name] * weight
                virtual_marker_data[vm_name] = vm_positions
            
            self._marker_names += list(virtual_marker_data.keys())
            self._trajectories.update(virtual_marker_data)

    @property
    def data(self):  # TODO: elsewhere this is used for the numpy array, but here its a dictionary 
        return self._trajectories

    @property
    def landmark_data(self):
        return {landmark_name:trajectory for landmark_name, trajectory in self._trajectories.items() if landmark_name in self._tracked_point_names}

    @property
    def virtual_marker_data(self):
        if not self._virtual_marker_definitions:
            return {}
        return {marker_name:trajectory for marker_name, trajectory in self._trajectories.items() if marker_name in self._virtual_marker_definitions.keys()}

    @property
    def segment_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        if not self._segment_connections:
            return {}
        
        segment_positions = {}
        for name, connection in self._segment_connections.items():
            proximal = self._trajectories.get(connection["proximal"])
            distal = self._trajectories.get(connection["distal"])

            segment_positions.update({name: {'proximal': proximal, 'distal': distal}})

        return segment_positions
    
    @property
    def tracked_point_names(self):
        """The original tracked point names excluding virtual markers"""
        return self._tracked_point_names
    
    @property
    def num_frames(self):
        return self._num_frames
    
    @property
    def as_numpy(self):
        numpy_data = np.full((self._num_frames, len(self._marker_names), 3), np.nan)

        for marker_idx, marker_name in enumerate(self._marker_names):
            if marker_name in self._trajectories:
                numpy_data[:, marker_idx, :] = self._trajectories[marker_name]

        return numpy_data
    
    @property
    def as_dataframe(self):
        tidy_data = []

        for frame_number in range(self._num_frames):
            frame_data = self.get_frame(frame_number)
            for marker_name, marker_3d_position in frame_data.items():
                
                x, y, z = marker_3d_position

                tidy_data.append({
                    "frame": frame_number,
                    "keypoint": marker_name,
                    "x": x,
                    "y": y,
                    "z": z
                })

        return pd.DataFrame(tidy_data)        


    def get_marker(self, marker_name: str):
        return self._trajectories[marker_name]

    def get_frame(self, frame_number: int):
        return {marker_name: trajectory[frame_number] for marker_name, trajectory in self._trajectories.items()}

    def __str__(self) -> str:

        return f"Trajectory with {self._num_frames} frames and {len(self._marker_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()

