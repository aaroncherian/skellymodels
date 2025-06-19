from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import Dict, List
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection
from skellymodels.models.anatomical_structure import AnatomicalStructure
import warnings
class Trajectory(BaseModel):
    name: str
    array: np.ndarray
    landmark_names: List[MarkerName]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_tracked_points_data(cls, 
                                 name:str,
                                tracked_points_array:np.ndarray, 
                                anatomical_structure:AnatomicalStructure):
        landmark_names = anatomical_structure.tracked_point_names.copy()
        vm_defs = anatomical_structure.virtual_markers_definitions
        
        output_array_as_list: list[np.ndarray] = [tracked_points_array]
        #compute virtual markers
        if vm_defs:
            for vm_name, vm_components in vm_defs.items():
                component_names = vm_components["marker_names"]
                component_weights = vm_components["marker_weights"]

                component_indices = [anatomical_structure.tracked_point_names.index(name) for name in component_names]
                component_data = tracked_points_array[:,component_indices,:]
                component_weights = np.array(component_weights)[None, :, None]
                weighted_marker_data = component_data * component_weights
                virtual_marker = np.sum(weighted_marker_data, axis = 1)
                output_array_as_list.append(virtual_marker[:, None, :])
                landmark_names.append(vm_name) 

        output_array = np.concatenate(output_array_as_list, axis=1) if len(output_array_as_list) > 1 else tracked_points_array
    
        return cls(name = name, 
                   array = output_array,
                   landmark_names = landmark_names)
    
    @model_validator(mode="after")
    def _check_shape(self):
        if self.array.shape[1] != len(self.landmark_names):
            raise ValueError(
                f"{self.name}: data has {self.array.shape[1]} columns but "
                f"{len(self.landmark_names)} marker names supplied"
            )
        if self.array.shape[2] != 3:
            raise ValueError(f"{self.name}: last dim must be 3 (xyz)")
        return self

    @property
    def as_array(self) -> np.ndarray:
        return self.array
    
    @property
    def as_dict(self) -> dict[MarkerName, np.ndarray]:
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
    
    def segment_data(self, segment_connections:Dict[SegmentName, SegmentConnection]) -> Dict[SegmentName, Dict[str, np.ndarray]]:
        if not segment_connections:
            return {}
        d = self.as_dict
        segment_positions = {}
        for name, connection in segment_connections.items():
            proximal = d.get(connection["proximal"])
            distal = d.get(connection["distal"])

            segment_positions.update({name: {'proximal': proximal, 'distal': distal}})

        return segment_positions
    
    @property
    def data(self): #this is for me to figure out where I use this in my validation pipeline. Will remove this after that.
        warnings.warn(".data is deprecated - use .as_dict for the same output",
                      DeprecationWarning,
                      stacklevel=2)  # TODO: elsewhere this is used for the numpy array, but here its a dictionary 
        return self.as_dict

    
    def __str__(self) -> str:
        return f"Trajectory with {self.num_frames} frames and {len(self.landmark_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()
