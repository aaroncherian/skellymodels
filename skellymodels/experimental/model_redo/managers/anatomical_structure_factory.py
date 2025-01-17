
from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure 
from skellymodels.experimental.model_redo.builders.anatomical_structure_builder import AnatomicalStructureBuilder

from typing import Dict, Optional
class AnatomicalStructureFactory:
    def __init__(self, model_info):
        self.model_info = model_info

    def create_structures(self) -> Dict[str, Optional[AnatomicalStructure]]:
        structures = {}
        for aspect_name, aspect_structure in self.model_info.aspects.items():

            structures[aspect_name] = (AnatomicalStructureBuilder()
                .with_tracked_points(aspect_structure.tracked_points_names)
                .with_virtual_markers(aspect_structure.virtual_marker_definitions)
                .with_segment_connections(aspect_structure.segment_connections)
                .with_center_of_mass(aspect_structure.center_of_mass_definitions)
                .with_joint_hierarchy(aspect_structure.joint_hierarchy)
            ).build()

        return structures

from skellymodels.experimental.model_redo.tracker_info.model_info import MediaPipeModelInfo

def create_anatomical_structure_factory(tracker_name: str) -> AnatomicalStructureFactory:
    tracker_info = {
        "mediapipe": MediaPipeModelInfo(),
    }
    
    model_info = tracker_info.get(tracker_name)
    if model_info is None:
        raise ValueError(f"Unsupported tracker type: {tracker_name}")
        
    return AnatomicalStructureFactory(model_info)