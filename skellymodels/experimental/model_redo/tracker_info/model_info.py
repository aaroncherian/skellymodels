from pathlib import Path
from typing import Union, Dict, List, Optional
import yaml
from dataclasses import dataclass

@dataclass
class AspectInfo:
    tracked_points_names: List[str]
    virtual_marker_definitions: Optional[Dict[str, Dict[str, List[Union[float, str]]]]] = None
    segment_connections: Optional[Dict[str, Dict[str, str]]] = None
    center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None
    joint_hierarchy: Optional[Dict[str, List[str]]] = None

class ModelInfo:
    def __init__(self, config_path: Union[str, Path]):
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.name = config['name']
        self.tracker_name = config['tracker_name']

        self.aspects: Dict[str, Dict ] = {}

        for aspect_name, aspect_info in config['aspects'].items():
            #handle tracked points names based on method specified in yaml
            if aspect_info['tracked_points']['type'] == 'list':
                tracked_points_names = aspect_info['tracked_points']['names']
            elif aspect_info['tracked_points']['type'] == 'pattern':
                naming_convention = aspect_info['tracked_points']['names']['convention']
                count = aspect_info['tracked_points']['names']['count']
                tracked_points_names = [naming_convention.format(i) for i in range(count)]
            
            self.aspects[aspect_name] = AspectInfo(
                tracked_points_names = tracked_points_names,
                virtual_marker_definitions = aspect_info.get('virtual_marker_definitions'),
                segment_connections = aspect_info.get('segment_connections'),
                center_of_mass_definitions = aspect_info.get('center_of_mass_definitions'),
                joint_hierarchy = aspect_info.get('joint_hierarchy')
            )

class MediaPipeModelInfo(ModelInfo):
    def __init__(self):
        super().__init__(config_path = Path.cwd()/'skellymodels'/'experimental'/'model_redo'/'tracker_info'/'mediapipe_info.yaml')





f = 2
