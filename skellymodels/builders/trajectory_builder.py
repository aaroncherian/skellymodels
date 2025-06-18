from skellymodels.utils.types import MarkerName, VirtualMarkerDefinition, SegmentConnection, SegmentName
from skellymodels.models.trajectory import Trajectory
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from typing import List, Dict

class TrajectoryValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data:np.ndarray
    tracked_point_names: List[str]

    @model_validator(mode="after")
    def validate_data(self):
        if self.data.shape[1] != len(self.tracked_point_names):
            raise ValueError(f"Trajectory data must have the same number of markers as input name list. Data has shape {self.data.shape} and list has {len(self.tracked_point_names)} markers.")
        
class TrajectoryBuilder:
    def __init__(self,
                tracked_point_names: List[MarkerName],
                virtual_marker_definitions: Dict[str, VirtualMarkerDefinition]|None = None,
                segment_connections: Dict[SegmentName, SegmentConnection]|None = None):
        self.tracked_point_names = tracked_point_names
        self.virtual_marker_definitions = virtual_marker_definitions
        self.segment_connections = segment_connections
    
    def _validate_data(self, data: np.ndarray, tracked_point_names: List[MarkerName]):
        TrajectoryValidator(data=data, tracked_point_names= tracked_point_names)

    def build(
            self,
            name: str,
            data_array: np.ndarray
    ) -> Trajectory:
        landmark_names = self.tracked_point_names.copy()
        output_array_as_list: list[np.ndarray] = [data_array]

        if self.virtual_marker_definitions:
            for vm_name, vm_components in self.virtual_marker_definitions.items():
                component_names = vm_components["marker_names"]
                component_weights = vm_components["marker_weights"]

                component_indices = [self.tracked_point_names.index(name) for name in component_names]
                component_data = data_array[:,component_indices,:]
                component_weights = np.array(component_weights)[None, :, None]
                weighted_marker_data = component_data * component_weights
                virtual_marker = np.sum(weighted_marker_data, axis = 1)
                output_array_as_list.append(virtual_marker[:, None, :])
                landmark_names.append(vm_name) 

        output_array = np.concatenate(output_array_as_list, axis=1) if len(output_array_as_list) > 1 else data_array

        return Trajectory(
            array = output_array,
            name = name,
            landmark_names = landmark_names,
            tracked_point_names = self.tracked_point_names,
            virtual_marker_definitions= self.virtual_marker_definitions,
            segment_connections = self.segment_connections,
    )