import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Union


@dataclass
class ModelInfo(dict):
    name: str
    tracker_name: str
    landmark_names: List[str]
    num_tracked_points: int
    tracked_object_names: Optional[list] = None
    virtual_markers_definitions: Optional[Dict[str, Dict[str, List[Union[str, float]]]]] = None
    segment_connections: Optional[Dict[str, Dict[str, str]]] = None
    center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None
    joint_hierarchy: Optional[Dict[str, List[str]]] = None


class MediapipeModel:
    def __init__(self):
        with open('skellymodels/experimental/model_info/mediapipe_config.json') as f:
            model_info = json.l1
        self.name = model_info['name']
        self.aspects = {}

    def _create_models_for_aspects(self, model_info):
        for aspect_name, aspect_info in model_info['aspects'].items():
            self.aspects[aspect_name] = ModelInfo(
                tracker_name=aspect_info['tracker_name'],
                landmark_names=aspect_info['landmark_names'],
                num_tracked_points=aspect_info['num_tracked_points'],
                tracked_object_names=aspect_info['tracked_object_names'],
                virtual_markers_definitions=aspect_info['virtual_markers_definitions'],
                segment_connections=aspect_info['segment_connections'],
                center_of_mass_definitions=aspect_info['center_of_mass_definitions'],
                joint_hierarchy=aspect_info['joint_hierarchy']
            )

b

mediapipe_model = MediapipeModel()
f = 2