from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure
from skellymodels.experimental.model_redo.models.trajectory import Trajectory

from typing import Dict, Any
import numpy as np

class Aspect:
    def __init__(self, name:str):
        self.name = name
        self.anatomical_structure = {}
        self.trajectories = {}
        self.metadata = {}

    def add_anatomical_structure(self, anatomical_structure: AnatomicalStructure):
        self.anatomical_structure = anatomical_structure

    def add_trajectories(self, name:str, trajectory:np.ndarray):
        """Add a complete set of trajectories including all markers"""
        self.trajectories[name] = Trajectory(name=name,
                                       data=trajectory,
                                       marker_names = self.anatomical_structure.marker_names,
                                       virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions)

    def add_landmark_trajectories(self, trajectory: np.ndarray):
        """Add trajectories for basic landmarks, calculating virtual markers if defined"""
        self.trajectories['main'] = Trajectory(name="main",
                                       data=trajectory,
                                       marker_names = self.anatomical_structure.landmark_names,
                                       virtual_marker_definitions=self.anatomical_structure.virtual_markers_definitions)

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
