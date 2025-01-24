from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.calculate_center_of_mass import calculate_center_of_mass_from_trajectory
from skellymodels.experimental.model_redo.fmc_anatomical_pipeline.enforce_rigid_bones import enforce_rigid_bones_from_trajectory
from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure
from skellymodels.experimental.model_redo.models.trajectory import Trajectory

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

    def __init__(self, name:str, anatomical_structure: Optional[AnatomicalStructure] = None, trajectories: Optional[Dict[str, Trajectory]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.anatomical_structure: Optional[AnatomicalStructure] = anatomical_structure
        self.trajectories: Dict[str, Trajectory] = {} if trajectories is None else trajectories  # TODO: the string keys make this clunky to use. If we "expect" to have 3d_xyz, total_body_com, and segment_com, could use an Enum
        self.metadata: Dict[str, Any] = {} if metadata is None else metadata  # TODO: Is it worth making a data class for this? Will we always want tracker type named or is it optional?

    def add_anatomical_structure(self, anatomical_structure: AnatomicalStructure):
        # TODO: should this be added after initialization? or will we ever intialize without an anatomical structure?
        self.anatomical_structure = anatomical_structure

    def add_trajectory(self, name:str, 
                         data:np.ndarray, 
                         marker_names:List[str],  # it looks like this can't actually be None, it will fail the Trajectory Validator
                         virtual_marker_definitions:Dict | None = None,
                         segment_connections:Dict | None = None):
        """Add a trajectory to the aspect"""
        self.trajectories[name] = Trajectory(name=name,
                                       data=data,
                                       marker_names = marker_names,
                                       virtual_marker_definitions=virtual_marker_definitions,
                                       segment_connections=segment_connections)

    def add_tracked_points(self, tracked_points: np.ndarray):
        """Use tracked points to calculate trajectories, using virtual markers if included"""
        if self.anatomical_structure is None or self.anatomical_structure.tracked_point_names is None:
            raise ValueError("Anatomical structure and tracked point names are required to add tracked points.")
        self.trajectories['3d_xyz'] = Trajectory(name="3d_xyz",
                                       data=tracked_points,
                                       marker_names = self.anatomical_structure.tracked_point_names,
                                       virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions,
                                       segment_connections=self.anatomical_structure.segment_connections)

    def add_total_body_center_of_mass(self, total_body_center_of_mass: np.ndarray):
        self.trajectories['total_body_com'] = Trajectory(name = 'total_body_com',
                                                         data = total_body_center_of_mass,
                                                         marker_names = ['total_body_com']
                                                         )
        
    def add_segment_center_of_mass(self, segment_center_of_mass:np.ndarray):
        if self.anatomical_structure is None or self.anatomical_structure.center_of_mass_definitions is None:
            raise ValueError("Anatomical structure and center of mass definitions are required to add segment center of mass.")
        self.trajectories['segment_com'] = Trajectory(name = 'segment_com',
                                                      data = segment_center_of_mass,
                                                      marker_names = list(self.anatomical_structure.center_of_mass_definitions.keys()))

    def add_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

    def add_tracker_type(self, tracker_type:str):
        # TODO: Same with anatomical structure, are we ever adding this after initialization?
        self.add_metadata({"tracker_type": tracker_type})

    def calculate_center_of_mass(self):
        if self.anatomical_structure is not None and self.anatomical_structure.center_of_mass_definitions is not None:
            print('Calculating center of mass for aspect:', self.name)
            total_body_com, segment_com = calculate_center_of_mass_from_trajectory(self.trajectories['3d_xyz'], self.anatomical_structure.center_of_mass_definitions)

            self.add_total_body_center_of_mass(total_body_center_of_mass=total_body_com)
            self.add_segment_center_of_mass(segment_center_of_mass=segment_com)
    
        else:
            print(f'Missing center of mass definitions for aspect {self.name}, skipping center of mass calculation')  # should be a warning when we switch to logging

    def enforce_rigid_bones(self):
        if self.anatomical_structure is not None and self.anatomical_structure.joint_hierarchy is not None:
            print('Enforcing rigid bones for aspect:', self.name)
            rigid_bones = enforce_rigid_bones_from_trajectory(self.trajectories['3d_xyz'], self.anatomical_structure.joint_hierarchy)

            self.add_trajectory(name = 'rigid_3d_xyz',
                                    data = rigid_bones,
                                    marker_names = self.anatomical_structure.marker_names)
        else:
            print(f'Missing segment connections for aspect {self.name}, skipping rigid bone enforcement')

    def __str__(self):
        anatomical_info = (
            str(self.anatomical_structure) if self.anatomical_structure else "No anatomical structure"
        )
        trajectory_info = (
            f"{len(self.trajectories)} trajectories: {list(self.trajectories.keys())}"
            if self.trajectories else "No trajectories"
        )
        metadata_info = (
            f": {self.metadata}"
            if self.metadata else "No metadata"
        )
        return (f"Aspect: {self.name}\n"
                f"  Anatomical Structure:\n{anatomical_info}\n"
                f"  Trajectories: {trajectory_info}\n"
                f"  Metadata: {metadata_info}\n\n")
    
    def __repr__(self):
        return self.__str__()
