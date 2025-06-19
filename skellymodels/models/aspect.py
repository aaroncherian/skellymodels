from pydantic import BaseModel, Field, model_validator, ConfigDict

from skellymodels.builders.anatomical_structure_builder import create_anatomical_structure_from_model_info
from skellymodels.tracker_info.model_info import ModelInfo
from skellymodels.builders.trajectory_builder import TrajectoryBuilder
from skellymodels.models.anatomical_structure import AnatomicalStructure
from skellymodels.models.error import Error
from skellymodels.models.trajectory import Trajectory
from skellymodels.utils.types import  SegmentName
from typing import Dict, Any, Optional
import numpy as np
from enum import Enum

class TrajectoryNames(Enum):
    """Enum for common trajectory names used in aspects."""
    XYZ = '3d_xyz'
    RIGID_XYZ = 'rigid_3d_xyz'
    TOTAL_BODY_COM = 'total_body_com'
    SEGMENT_COM = 'segment_com'

class Aspect(BaseModel):
    name: str
    anatomical_structure: AnatomicalStructure
    trajectories: dict[str,Trajectory] = Field(default_factory=dict)
    reprojection_error: Optional[Error] = None
    metadata: dict[str,Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, #to allow for numpy in the model
        validate_assignment=True) #validates after the model is changed
    
    @classmethod
    def from_model_info(cls, name: str, model_info: ModelInfo, metadata: Optional[Dict[str, Any]] = None):
        """ Class method to create an Aspect from a ModelInfo instance
        
        Args:
            name (str): Identifier for the aspect (e.g., "body", "face", "left_hand")
            model_info (ModelInfo): ModelInfo class instance
            metadata (Optional[Dict[str, Any]]): Additional information about the aspect (include the 'tracker)
        """
        anatomical_structure_dict = create_anatomical_structure_from_model_info(model_info=model_info)
        return cls(name=name, anatomical_structure=anatomical_structure_dict[name], metadata=metadata)
    
    def add_tracked_points(self, tracked_points:np.ndarray):
        """Ingests tracked points data and adds it to the aspect as a trajectory"""
        if self.anatomical_structure is None or self.anatomical_structure.tracked_point_names is None:
            raise ValueError("Anatomical structure and tracked point names are required to ingest tracker data.")

        self.trajectories[TrajectoryNames.XYZ.value] = Trajectory.from_tracked_points_data(
            name = TrajectoryNames.XYZ.value,
            tracked_points_array=tracked_points,
            anatomical_structure=self.anatomical_structure
        )

    def add_trajectory(self, 
                       dict_of_trajectories: Dict[str, Trajectory]):
        """ Adds all trajectories from a dictionary to the aspect."""
        for name, trajectory in dict_of_trajectories.items():
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"Expected Trajectory instance for {name}, got {type(trajectory)}")
            self.trajectories.update({name: trajectory})
            #add check for whether the trajectory name is in the expected list (and make an expected enum list)

    def add_reprojection_error(self, reprojection_error_data: np.ndarray):
        # TODO: This could be a feature of the trajectory as well, but I'm leaning towards aspect taking care of it
        if self.trajectories.get(TrajectoryNames.XYZ.value) is not None:
            if reprojection_error_data.shape[0] != self.trajectories[TrajectoryNames.XYZ.value].num_frames:
                raise ValueError(
                    "First dimension of reprojection error must match the number of frames in the trajectory.")
            if reprojection_error_data.shape[1] != len(self.anatomical_structure.tracked_point_names):
                raise ValueError(
                    "Second dimension of reprojection error must match the number of landmark names in the trajectory.")

        self.reprojection_error = Error(name='reprojection_error',
                                        data=reprojection_error_data,
                                        marker_names=self.anatomical_structure.tracked_point_names)

    def add_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

    @property
    def xyz(self) -> Optional[Trajectory]:
        """Returns the 3D XYZ trajectory if it exists, otherwise None."""
        return self.trajectories.get(TrajectoryNames.XYZ.value)
    
    @property
    def rigid_xyz(self) -> Optional[Trajectory]:
        """Returns the rigid body 3D XYZ trajectory if it exists, otherwise None."""
        return self.trajectories.get(TrajectoryNames.RIGID_XYZ.value)

    @property
    def total_body_com(self) -> Optional[Trajectory]:
        """Returns the total body center of mass trajectory if it exists, otherwise None."""
        return self.trajectories.get(TrajectoryNames.TOTAL_BODY_COM.value)
    
    @property
    def segment_com(self) -> Optional[Dict[SegmentName, Trajectory]]:
        """Returns the segment center of mass trajectory if it exists, otherwise None."""
        return self.trajectories.get(TrajectoryNames.SEGMENT_COM.value)

    def __str__(self):
        anatomical_info = (
            str(self.anatomical_structure) if self.anatomical_structure else "No anatomical structure"
        )
        trajectory_info = (
            f"{len(self.trajectories)} trajectories: {list(self.trajectories.keys())}"
            if self.trajectories else "No trajectories"
        )
        error_info = (
            f"Has reprojection error"
            if self.reprojection_error else "No reprojection error"
        )
        metadata_info = (
            f": {self.metadata}"
            if self.metadata else "No metadata"
        )
        return (f"Aspect: {self.name}\n"
                f"  Anatomical Structure:\n{anatomical_info}\n"
                f"  Trajectories: {trajectory_info}\n"
                f"  Error: {error_info}\n"
                f"  Metadata: {metadata_info}\n\n")

    def __repr__(self):
        return self.__str__()

