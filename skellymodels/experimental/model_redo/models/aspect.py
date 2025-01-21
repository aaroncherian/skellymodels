from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure
from skellymodels.experimental.model_redo.models.trajectory import Trajectory

from typing import Dict, Any, List, Optional
import numpy as np

class Aspect:
    def __init__(self, name:str):
        self.name = name
        self.anatomical_structure: Optional[AnatomicalStructure] = None
        self.trajectories: Dict[str, Trajectory] = {}
        self.metadata: Dict[str, Any] = {}

    def add_anatomical_structure(self, anatomical_structure: AnatomicalStructure):
        self.anatomical_structure = anatomical_structure

    def add_trajectory(self, name:str, 
                         data:np.ndarray, 
                         marker_names:List[str] = None,
                         virtual_marker_definitions:Dict = None,
                         segment_connections:Dict = None):
        self.trajectories[name] = Trajectory(name=name,
                                       data=data,
                                       marker_names = marker_names,
                                       virtual_marker_definitions=virtual_marker_definitions,
                                       segment_connections=segment_connections)

    def add_tracked_points(self, tracked_points: np.ndarray):
        """Add trajectories for tracked points, calculating virtual markers if defined"""
        self.trajectories['3d_xyz'] = Trajectory(name="3d_xyz",
                                       data=tracked_points,
                                       marker_names = self.anatomical_structure.tracked_point_names,
                                       virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions,
                                       segment_connections=self.anatomical_structure.segment_connections)

    def add_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

    def add_tracker_type(self, tracker_type:str):
        self.add_metadata({"tracker_type": tracker_type})

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
