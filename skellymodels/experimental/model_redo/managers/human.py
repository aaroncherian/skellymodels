from enum import Enum
import numpy as np

from skellymodels.experimental.model_redo.models.aspect import Aspect
from skellymodels.experimental.model_redo.managers.actor import Actor
from skellymodels.experimental.model_redo.managers.anatomical_structure_factory import create_anatomical_structure

from dataclasses import dataclass

from skellymodels.experimental.model_redo.utils.create_mediapipe_actor import split_data
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
class HumanAspectNames(Enum):
    BODY = "body"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class Human(Actor):
    def __init__(self, name: str, model_info:ModelInfo):
        super().__init__(name)
        
        self.tracker = model_info.name
        self.aspect_order = model_info.aspect_order_and_slices
        self.structures = create_anatomical_structure(model_info=model_info)
        
        self._initialize_aspects()
        f = 2

    def _initialize_aspects(self):
        self._add_body()

        if HumanAspectNames.FACE.value in self.aspect_order.keys():
            self._add_face()

        if HumanAspectNames.LEFT_HAND.value in self.aspect_order.keys():
            self._add_left_hand()
        
        if HumanAspectNames.RIGHT_HAND.value in self.aspect_order.keys():
            self._add_right_hand()

    def _add_body(self):
        body = Aspect(name = HumanAspectNames.BODY.value)
        body.add_metadata({"tracker_type": self.tracker})
        body.add_anatomical_structure(self.structures[HumanAspectNames.BODY.value])
        self.add_aspect(body)
    
    def _add_face(self):
        face = Aspect(name = HumanAspectNames.FACE.value)
        face.add_metadata({"tracker_type": self.tracker})
        face.add_anatomical_structure(self.structures[HumanAspectNames.FACE.value])
        self.add_aspect(face)

    def _add_left_hand(self):
        left_hand = Aspect(name = HumanAspectNames.LEFT_HAND.value)
        left_hand.add_metadata({"tracker_type": self.tracker})
        left_hand.add_anatomical_structure(self.structures[HumanAspectNames.LEFT_HAND.value])
        self.add_aspect(left_hand)

    def _add_right_hand(self):
        right_hand = Aspect(name = HumanAspectNames.RIGHT_HAND.value)
        right_hand.add_metadata({"tracker_type": self.tracker})
        right_hand.add_anatomical_structure(self.structures[HumanAspectNames.RIGHT_HAND.value])
        self.add_aspect(right_hand)

    @property
    def body(self):
        return self.aspects[HumanAspectNames.BODY.value]
    @property
    def face(self):
        return self.aspects.get(HumanAspectNames.FACE.value)
    @property
    def left_hand(self):
        return self.aspects.get(HumanAspectNames.LEFT_HAND.value)
    @property
    def right_hand(self):
        return self.aspects.get(HumanAspectNames.RIGHT_HAND.value)

    def from_tracked_points_numpy(self, tracked_points_numpy_array:np.ndarray):
        self.body.add_tracked_points(tracked_points_numpy_array[:,self.aspect_order[HumanAspectNames.BODY.value],:])

        if HumanAspectNames.FACE.value in self.aspect_order.keys():
            self.face.add_tracked_points(tracked_points_numpy_array[:,self.aspect_order[HumanAspectNames.FACE.value],:])

        if HumanAspectNames.LEFT_HAND.value in self.aspect_order.keys():
            self.left_hand.add_tracked_points(tracked_points_numpy_array[:,self.aspect_order[HumanAspectNames.LEFT_HAND.value],:])
        
        if HumanAspectNames.RIGHT_HAND.value in self.aspect_order.keys():
            self.right_hand.add_tracked_points(tracked_points_numpy_array[:,self.aspect_order[HumanAspectNames.RIGHT_HAND.value],:])


