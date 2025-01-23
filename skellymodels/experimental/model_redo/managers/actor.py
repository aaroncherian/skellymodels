from skellymodels.experimental.model_redo.models.aspect import Aspect
from typing import Dict

class Actor:
    """
    The Actor class is a container for multiple Aspects of a single person/creature/object that we track in 3D.
    """
    def __init__(self, name: str):
        self.name = name
        self.aspects: Dict[str, Aspect] = {}

    def __getitem__(self, key: str):
        return self.aspects[key]

    def __str__(self):
        return str(self.aspects.keys())

    def add_aspect(self, aspect: Aspect):
        self.aspects[aspect.name] = aspect

    def get_data(self, aspect_name:str, type:str):
        return self.aspects[aspect_name].trajectories[type].data

    def get_marker_data(self, aspect_name:str, type:str, marker_name:str):
        return self.aspects[aspect_name].trajectories[type].get_marker(marker_name)

    def get_frame(self, aspect_name:str, type:str, frame_number:int):
        return self.aspects[aspect_name].trajectories[type].get_frame(frame_number)


        