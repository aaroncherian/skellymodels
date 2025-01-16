from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure 
from skellymodels.experimental.model_redo.builders.anatomical_structure_builder import AnatomicalStructureBuilder

from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import MediapipeModelInfo

@dataclass
class HumanConfiguration:
    include_face: bool = True
    include_left_hand: bool = True
    include_right_hand: bool = True
    tracker_type: str = "mediapipe"


class AnatomicalStructureFactory(ABC):

    def __init__(self, configuration: HumanConfiguration):
        self.config = configuration

    @abstractmethod
    def create_body_structure(self) -> AnatomicalStructure:
        pass

    @abstractmethod
    def create_face_structure(self) -> AnatomicalStructure:
        pass

    @abstractmethod
    def create_left_hand_structure(self) -> AnatomicalStructure:
        pass

    @abstractmethod
    def create_right_hand_structure(self) -> AnatomicalStructure:
        pass

    def create_structures(self) -> Dict[str, Optional[AnatomicalStructure]]:
        structures = {
            "body": self.create_body_structure(),
        }

        if self.config.include_face:
            structures["face"] = self.create_face_structure()
        
        if self.config.include_hands:
            structures["left_hand"] = self.create_left_hand_structure()
            structures["right_hand"] = self.create_right_hand_structure()

        return structures
    
class MediaPipeStructureFactory(AnatomicalStructureFactory):

    def create_body_structure(self) -> AnatomicalStructure:
        return (AnatomicalStructureBuilder()
                .with_tracked_points(MediapipeModelInfo().body_landmark_names)
                .with_virtual_markers(MediapipeModelInfo().virtual_markers_definitions)
                .with_segment_connections(MediapipeModelInfo().segment_connections)
                .with_center_of_mass(MediapipeModelInfo().center_of_mass_definitions)
                .with_joint_hierarchy(MediapipeModelInfo().joint_hierarchy)
        ).build()
    
    def create_face_structure(self) -> AnatomicalStructure:
        return (AnatomicalStructureBuilder()
                .with_tracked_points([str(i).zfill(4) for i in range(MediapipeModelInfo().num_tracked_points_face)])
        ).build()
    
    def create_left_hand_structure(self) -> AnatomicalStructure:
        return (AnatomicalStructureBuilder()
                .with_tracked_points([f"left_{str(i).zfill(4)}" for i in range(MediapipeModelInfo().num_tracked_points_left_hand)])
        ).build()
    
    def create_right_hand_structure(self) -> AnatomicalStructure:
        return (AnatomicalStructureBuilder()
                .with_tracked_points([f"right_{str(i).zfill(4)}" for i in range(MediapipeModelInfo().num_tracked_points_right_hand)])
        ).build()


def create_anatomical_structure_factory(configuration: HumanConfiguration) -> AnatomicalStructureFactory:
    trackers = {
        "mediapipe": MediaPipeStructureFactory,
    }
    
    factory_class = trackers.get(configuration.tracker_type)
    if factory_class is None:
        raise ValueError(f"Unsupported tracker type: {configuration.tracker_type}")
        
    return factory_class(configuration)