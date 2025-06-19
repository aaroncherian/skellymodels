from enum import Enum
from skellymodels.managers.actor import Actor
from skellymodels.tracker_info.model_info import ModelInfo
from skellymodels.models.aspect import Aspect
import numpy as np

class AnimalAspectName(Enum):
    BODY = "body"

class Animal(Actor):
    def __init__(self, name: str, model_info:ModelInfo):
        super().__init__(name, model_info)
        self._add_body()

    def _add_body(self):
        self.aspect_from_model_info(
            name = AnimalAspectName.BODY.value
            )
        
    @property
    def body(self) -> Aspect:
        return self.aspects[AnimalAspectName.BODY.value]

    def add_tracked_points_numpy(self, tracked_points_numpy_array:np.ndarray):
        self.body.add_tracked_points(
            tracked_points_numpy_array[:,self.tracked_point_slices[AnimalAspectName.BODY.value],:]
        )

    def add_reprojection_error_numpy(self, reprojection_error_data: np.ndarray):
        self.body.add_reprojection_error(reprojection_error_data[:, self.tracked_point_slices[AnimalAspectName.BODY.value]])
