from enum import Enum
import numpy as np
from skellymodels.models.aspect import Aspect
from skellymodels.managers.animal import Animal
from skellymodels.tracker_info.model_info import ModelInfo
class HumanAspectNames(Enum):
    BODY = "body"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"

HUMAN_ENUMS = (HumanAspectNames.FACE,
                HumanAspectNames.LEFT_HAND,
                HumanAspectNames.RIGHT_HAND)

class Human(Animal):
    def __init__(self, name: str, model_info:ModelInfo):
        super().__init__(name, model_info)
    
        self._initialize_aspects()

    def _initialize_aspects(self):
        """
        Initializes the predefined anatomical aspects (body, face, hands) for the Human instance.
        Aspects are added based on the configuration provided in the ModelInfo instance.
        """

        for aspect_enum in HUMAN_ENUMS:
            if aspect_enum.value in self.aspect_order:
                self.aspect_from_model_info(
                    name = aspect_enum.value
                )
    
    @property
    def face(self) -> Aspect:
        return self.aspects.get(HumanAspectNames.FACE.value)
    @property
    def left_hand(self) -> Aspect:
        return self.aspects.get(HumanAspectNames.LEFT_HAND.value)
    @property
    def right_hand(self) -> Aspect:
        return self.aspects.get(HumanAspectNames.RIGHT_HAND.value)
    
    def add_tracked_points_numpy(self, tracked_points_numpy_array:np.ndarray):
        """
        Takes in the tracked points array, splits and categorizes it based on the ranges determined by the ModelInfo,
        and adds it as a tracked point Trajectory to the body, and optionally face/hands aspects 
        """

        super().add_tracked_points_numpy(tracked_points_numpy_array)

        for aspect_enum in HUMAN_ENUMS:
            
            aspect = self.aspects.get(aspect_enum.value)
            if aspect_enum.value in self.tracked_point_slices and aspect:
                aspect.add_tracked_points(
                    tracked_points_numpy_array[:, self.tracked_point_slices[aspect_enum.value],:]
                )

    def add_reprojection_error_numpy(self, reprojection_error_data: np.ndarray):
        
        super().add_reprojection_error_numpy(reprojection_error_data)

        for aspect_enum in HUMAN_ENUMS:
            
            aspect = self.aspects.get(aspect_enum.value)
            if aspect_enum.value in self.tracked_point_slices and aspect:
                aspect.add_reprojection_error(
                    reprojection_error_data[:, self.tracked_point_slices[aspect_enum.value]]
                )
