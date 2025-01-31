from skellymodels.experimental.model_redo.builders.anatomical_structure_builder import create_anatomical_structure_from_model_info
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo

from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure
from skellymodels.experimental.model_redo.models.error import Error
from skellymodels.experimental.model_redo.models.trajectory import Trajectory

# from skellymodels.experimental.model_redo.biomechanics.biomechanics_wrappers import (
#     calculate_center_of_mass,
#     enforce_rigid_bones_from_trajectory,
# )

from typing import Dict, Any, List, Optional
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

    def __init__(self, name: str, anatomical_structure: AnatomicalStructure,
                 trajectories: Optional[Dict[str, Trajectory]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.anatomical_structure = anatomical_structure
        self.trajectories: Dict[
            str, Trajectory] = {} if trajectories is None else trajectories  # TODO: the string keys make this clunky to use. If we "expect" to have 3d_xyz, total_body_com, and segment_com, could use an Enum
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

    def add_trajectory(self, name: str,
                       data: np.ndarray,
                       marker_names: List[str],
                       virtual_marker_definitions: Dict | None = None,
                       segment_connections: Dict | None = None):
        """Add a trajectory to the aspect"""
        self.trajectories[name] = Trajectory(name=name,
                                             data=data,
                                             marker_names=marker_names,
                                             virtual_marker_definitions=virtual_marker_definitions,
                                             segment_connections=segment_connections)

    def add_tracked_points(self, tracked_points: np.ndarray):
        """Use tracked points to calculate trajectories, using virtual markers if included"""
        if self.anatomical_structure is None or self.anatomical_structure.tracked_point_names is None:
            raise ValueError("Anatomical structure and tracked point names are required to add tracked points.")
        self.trajectories['3d_xyz'] = Trajectory(name="3d_xyz",
                                                 data=tracked_points,
                                                 marker_names=self.anatomical_structure.tracked_point_names,
                                                 virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions,
                                                 segment_connections=self.anatomical_structure.segment_connections)

    def add_total_body_center_of_mass(self, total_body_center_of_mass: np.ndarray):
        self.trajectories['total_body_com'] = Trajectory(name='total_body_com',
                                                         data=total_body_center_of_mass,
                                                         marker_names=['total_body_com']
                                                         )

    def add_segment_center_of_mass(self, segment_center_of_mass: np.ndarray):
        if self.anatomical_structure is None or self.anatomical_structure.center_of_mass_definitions is None:
            raise ValueError(
                "Anatomical structure and center of mass definitions are required to add segment center of mass.")
        self.trajectories['segment_com'] = Trajectory(name='segment_com',
                                                      data=segment_center_of_mass,
                                                      marker_names=list(
                                                          self.anatomical_structure.center_of_mass_definitions.keys()))

    def add_reprojection_error(self, reprojection_error_data: np.ndarray):
        # TODO: This could be a feature of the trajectory as well, but I'm leaning towards aspect taking care of it
        if self.trajectories.get('3d_xyz') is not None:
            if reprojection_error_data.shape[0] != self.trajectories['3d_xyz'].num_frames:
                raise ValueError(
                    "First dimension of reprojection error must match the number of frames in the trajectory.")
            if reprojection_error_data.shape[1] != len(self.trajectories['3d_xyz'].tracked_point_names):
                raise ValueError(
                    "Second dimension of reprojection error must match the number of landmark names in the trajectory.")

        self.reprojection_error = Error(name='reprojection_error',
                                        data=reprojection_error_data,
                                        marker_names=self.trajectories['3d_xyz'].tracked_point_names)

    def add_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

    def add_tracker_type(self, tracker_type: str):
        # TODO: Same with anatomical structure, are we ever adding this after initialization?
        self.add_metadata({"tracker_type": tracker_type})

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
