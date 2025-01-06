from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
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
