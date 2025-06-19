from enum import Enum
from skellymodels.managers.actor import Actor
from skellymodels.tracker_info.model_info import ModelInfo
from skellymodels.models.aspect import Aspect


class AnimalAspectName(Enum):
    BODY = "body"

class Animal(Actor):
    def __init__(self, name: str, model_info:ModelInfo):
        super().__init__(name, model_info)
        self._add_body()

    def _add_body(self):
        body = Aspect.from_model_info(
            name = AnimalAspectName.BODY.value,
            model_info = self.model_info,
            metadata = {"tracker_type": self.tracker}
        )
        self.add_aspect(body)

    @property
    def body(self) -> Aspect:
        return self.aspects[AnimalAspectName.BODY.value]

    def add_tracked_points_numpy(self, tracked_points_numpy_array):
        self.body.add_tracked_points(
            tracked_points_numpy_array[:,self.tracked_point_slices[AnimalAspectName.BODY.value],:]
        )
