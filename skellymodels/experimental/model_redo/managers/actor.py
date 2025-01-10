from skellymodels.experimental.model_redo.models.aspect import Aspect
from skellymodels.experimental.model_redo.managers.anatomical_structure_factory import create_anatomical_structure_factory

from dataclasses import dataclass


class Actor:
    def __init__(self, name: str):
        self.name = name
        self.aspects = {}

    def __getitem__(self, key: str):
        return self.aspects[key]

    def __str__(self):
        return str(self.aspects.keys())

    def add_aspect(self, aspect: Aspect):
        self.aspects[aspect.name] = aspect

    def get_data(self, aspect_name:str, type:str):
        return self.aspects[aspect_name].trajectories[type].trajectories

    def get_marker_data(self, aspect_name:str, type:str, marker_name:str):
        return self.aspects[aspect_name].trajectories[type].get_marker(marker_name)

    def get_frame(self, aspect_name:str, type:str, frame_number:int):
        return self.aspects[aspect_name].trajectories[type].get_frame(frame_number)

from enum import Enum

class HumanAspects(Enum):
    BODY = "body"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"

@dataclass
class HumanConfiguration:
    include_face: bool = True
    include_left_hand: bool = True
    include_right_hand: bool = True
    tracker_type: str = "mediapipe"

class Human(Actor):
    def __init__(self, name: str, configuration: HumanConfiguration):
        super().__init__(name)
        self.config = configuration

        self.structures = create_anatomical_structure_factory(self.config).create_structures()
        
        self._initialize_aspects()

    def _initialize_aspects(self):
        self._add_body()

        if self.config.include_face:
            self._add_face()

        if self.config.include_left_hand:
            self._add_left_hand()

        if self.config.include_right_hand:
            self._add_right_hand()

    def _add_body(self):
        body = Aspect(name = HumanAspects.BODY.value)
        body.add_metadata({"tracker_type": "mediapipe"})
        body.add_anatomical_structure(self.structures["body"])
        self.add_aspect(body)
    
    def _add_face(self):
        face = Aspect(name = HumanAspects.FACE.value)
        face.add_metadata({"tracker_type": "mediapipe"})
        face.add_anatomical_structure(self.structures["face"])
        self.add_aspect(face)

    def _add_left_hand(self):
        left_hand = Aspect(name = HumanAspects.LEFT_HAND.value)
        left_hand.add_metadata({"tracker_type": "mediapipe"})
        left_hand.add_anatomical_structure(self.structures["left_hand"])
        self.add_aspect(left_hand)

    def _add_right_hand(self):
        right_hand = Aspect(name = HumanAspects.RIGHT_HAND.value)
        right_hand.add_metadata({"tracker_type": "mediapipe"})
        right_hand.add_anatomical_structure(self.structures["right_hand"])
        self.add_aspect(right_hand)

    @property
    def body(self):
        return self.aspects[HumanAspects.BODY.value]
    
    @property
    def face(self):
        return self.aspects.get(HumanAspects.FACE.value)
    @property
    def left_hand(self):
        return self.aspects.get(HumanAspects.LEFT_HAND.value)
    
    @property
    def right_hand(self):
        return self.aspects.get(HumanAspects.RIGHT_HAND.value)
    


    
        


human = Human(name="human_one", configuration=HumanConfiguration(include_face=False, include_left_hand=True, include_right_hand=True))
f = 2