from skellymodels.builders.anatomical_structure_builder import create_anatomical_structure_from_model_info
from skellymodels.tracker_info.model_info import ModelInfo
from skellymodels.builders.trajectory_builder import TrajectoryBuilder
from skellymodels.models.anatomical_structure import AnatomicalStructure
from skellymodels.models.error import Error
from skellymodels.models.trajectory import Trajectory
from skellymodels.utils.types import  SegmentName
from typing import Dict, Any, Optional
import numpy as np


class Aspect:
    """
    An Aspect represents a distinct component of a tracked object (e.g., body, face, hand) containing 
    the anatomical structure and 3D data for that tracked object. It handles structural definitions (through AnatomicalStructure), 
    and using those definitions to turn 3d data into Trajectories.

    Attributes:
        name (str): Identifier for the aspect (e.g., "body", "face", "left_hand")
        anatomical_structure (Optional[AnatomicalStructure]): Defines the structural properties like 
            landmarks, virtual markers, segments, and center of mass definitions
        trajectories (Dict[str, Trajectory]): Collection of named trajectory data sets
        metadata (Dict[str, Any]): Additional information about the aspect
    """

    def __init__(self, name: str, 
                 anatomical_structure: AnatomicalStructure,
                 trajectories: Optional[Dict[str, Trajectory]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.name = name
        self.anatomical_structure = anatomical_structure
        self.trajectories: Dict[
            str, Trajectory] = {} if trajectories is None else trajectories 
        self.reprojection_error: Optional[Error] = None
        self.metadata: Dict[
            str, Any] = {} if metadata is None else metadata  # TODO: Is it worth making a data class for this? Will we always want tracker type named or is it optional?

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

        builder = TrajectoryBuilder(
            tracked_point_names=self.anatomical_structure.tracked_point_names,
            virtual_marker_definitions= self.anatomical_structure.virtual_markers_definitions,
            segment_connections= self.anatomical_structure.segment_connections
        )

        self.trajectories['3d_xyz'] = builder.build(
            name='3d_xyz',
            data_array=tracked_points
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
        if self.trajectories.get('3d_xyz') is not None:
            if reprojection_error_data.shape[0] != self.trajectories['3d_xyz'].num_frames:
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
        return self.trajectories.get('3d_xyz')
    
    @property
    def rigid_xyz(self) -> Optional[Trajectory]:
        """Returns the rigid body 3D XYZ trajectory if it exists, otherwise None."""
        return self.trajectories.get('rigid_3d_xyz')

    @property
    def total_body_com(self) -> Optional[Trajectory]:
        """Returns the total body center of mass trajectory if it exists, otherwise None."""
        return self.trajectories.get('total_body_com')
    
    @property
    def segment_com(self) -> Optional[Dict[SegmentName, Trajectory]]:
        """Returns the segment center of mass trajectory if it exists, otherwise None."""
        return self.trajectories.get('segment_com')

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
