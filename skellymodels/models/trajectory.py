from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import Dict, List
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection
import warnings

class TrajectoryValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data:np.ndarray
    tracked_point_names: List[str]

    @model_validator(mode="after")
    def validate_data(self):
        if self.data.shape[1] != len(self.tracked_point_names):
            raise ValueError(f"Trajectory data must have the same number of markers as input name list. Data has shape {self.data.shape} and list has {len(self.tracked_point_names)} markers.")

class Trajectory(BaseModel):
    name: str
    array: np.ndarray
    tracked_point_names: List[MarkerName]
    landmark_names: List[MarkerName]
    virtual_marker_definitions: Dict[str, VirtualMarkerDefinition] | None = None
    segment_connections: Dict[SegmentName, SegmentConnection] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def as_array(self) -> np.ndarray:
        return self.array
    
    @property
    def as_dict(self) -> dict[str, np.ndarray]:
        return {n: self.array[:, i, :]
            for i, n in enumerate(self.landmark_names)}

    @property
    def as_dataframe(self) -> pd.DataFrame:

        df = pd.DataFrame(
            self.array.reshape(self.num_frames*self.num_markers,3),
            columns=['x','y','z'],
        )
        
        df['frame'] = np.repeat(np.arange(self.num_frames),self.num_markers)
        df['keypoint'] = np.tile(self.landmark_names, self.num_frames)
        return df[['frame', 'keypoint', 'x', 'y', 'z']]
         
    @property
    def num_frames(self) -> int:
        return self.array.shape[0]
    
    @property
    def num_markers(self) -> int:
        return self.array.shape[1]
    
    @property
    def landmark_names(self) -> list[MarkerName]:
        return self.landmark_names
    
    @property
    def tracked_point_names(self) -> list[MarkerName]:
        return self.tracked_point_names
    
    @property
    def segment_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        if not self.segment_connections:
            return {}
        d = self.as_dict
        segment_positions = {}
        for name, connection in self.segment_connections.items():
            proximal = d.get(connection["proximal"])
            distal = d.get(connection["distal"])

            segment_positions.update({name: {'proximal': proximal, 'distal': distal}})

        return segment_positions
    
    @property
    def data(self):
        warnings.warn(".data is deprecated - use .as_dict for the same output",
                      DeprecationWarning,
                      stacklevel=2)  # TODO: elsewhere this is used for the numpy array, but here its a dictionary 
        return self.as_dict

    
    def __str__(self) -> str:
        return f"Trajectory with {self.num_frames} frames and {len(self.landmark_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()

    # @property
    # def landmark_names(self):
    #     return self.landmark_names

    # @property
    # def data(self):
    #     warnings.warn(".data is deprecated - use .as_dict for the same output",
    #                   DeprecationWarning,
    #                   stacklevel=2)  # TODO: elsewhere this is used for the numpy array, but here its a dictionary 
    #     return self._trajectories

    # @property
    # def virtual_marker_data(self):
    #     if not self._virtual_marker_definitions:
    #         return {}
    #     return {marker_name:trajectory for marker_name, trajectory in self._trajectories.items() if marker_name in self._virtual_marker_definitions.keys()}

    # @property
    # def segment_data(self) -> Dict[str, Dict[str, np.ndarray]]:
    #     if not self._segment_connections:
    #         return {}
        
    #     segment_positions = {}
    #     for name, connection in self._segment_connections.items():
    #         proximal = self._trajectories.get(connection["proximal"])
    #         distal = self._trajectories.get(connection["distal"])

    #         segment_positions.update({name: {'proximal': proximal, 'distal': distal}})

    #     return segment_positions
    
    # @property
    # def tracked_point_names(self):
    #     """The original tracked point names excluding virtual markers"""
    #     return self._tracked_point_names
    
    # @property
    # def num_frames(self):
    #     return self._num_frames
    
    # @property
    # def as_numpy(self):
    #     numpy_data = np.full((self._num_frames, len(self._marker_names), 3), np.nan)

    #     for marker_idx, marker_name in enumerate(self._marker_names):
    #         if marker_name in self._trajectories:
    #             numpy_data[:, marker_idx, :] = self._trajectories[marker_name]

    #     return numpy_data
    
    # @property
    # def as_dataframe(self):
    #     tidy_data = []

    #     for frame_number in range(self._num_frames):
    #         frame_data = self.get_frame(frame_number)
    #         for marker_name, marker_3d_position in frame_data.items():
                
    #             x, y, z = marker_3d_position

    #             tidy_data.append({
    #                 "frame": frame_number,
    #                 "keypoint": marker_name,
    #                 "x": x,
    #                 "y": y,
    #                 "z": z
    #             })

    #     return pd.DataFrame(tidy_data)        


    # def get_marker(self, marker_name: str):
    #     return self._trajectories[marker_name]

    # def get_frame(self, frame_number: int):
    #     return {marker_name: trajectory[frame_number] for marker_name, trajectory in self._trajectories.items()}

    # def __str__(self) -> str:

    #     return f"Trajectory with {self._num_frames} frames and {len(self._marker_names)} markers"

    # def __repr__(self) -> str:
    #     return self.__str__()


class TrajectoryBuilder:

    def __init__(self,
                name: str, 
                tracked_point_names: List[MarkerName],
                virtual_marker_definitions: Dict[str, VirtualMarkerDefinition]|None = None,
                segment_connections: Dict|None = None):
        self.name = name
        self.tracked_point_names = tracked_point_names
        self.virtual_marker_definitions = virtual_marker_definitions
        self.segment_connections = segment_connections
    
    def _validate_data(self, data: np.ndarray, tracked_point_names: List[MarkerName]):
        TrajectoryValidator(data=data, tracked_point_names= tracked_point_names)

    def build(
            self,
            data_array: np.ndarray
    ):
        landmark_names = self.tracked_point_names.copy()
        output_array_as_list: list[np.ndarray] = [data_array]

        if self.virtual_marker_definitions:
            for vm_name, vm_components in self.virtual_marker_definitions.items():
                component_names = vm_components["marker_names"]
                component_weights = vm_components["marker_weights"]

                component_indices = [self.tracked_point_names.index(name) for name in component_names]
                component_data = data_array[:,component_indices,:]
                component_weights = component_weights[None, :, None]
                weighted_marker_data = component_data * component_weights
                virtual_marker = np.sum(weighted_marker_data, axis = 1)
                output_array_as_list.append(virtual_marker[:, None, :])
                landmark_names.append(vm_name) 

        output_array = np.concatenate(output_array_as_list, axis=1) if len(output_array_as_list) > 1 else data_array

        return Trajectory(
            array = output_array,
            name = self.name,
            landmark_names = landmark_names,
            tracked_point_names = self.tracked_point_names,
            virtual_marker_definitions= self.virtual_marker_definitions,
            segment_connections = self.segment_connections,
    )