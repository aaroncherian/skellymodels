from enum import Enum
import numpy as np
import pandas as pd

from skellymodels.models.aspect import Aspect
from skellymodels.models.trajectory import Trajectory
from skellymodels.managers.actor import Actor
from skellymodels.tracker_info.model_info import ModelInfo
class HumanAspectNames(Enum):
    BODY = "body"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class Human(Actor):
    def __init__(self, name: str, model_info:ModelInfo):
        super().__init__(name)
        
        self.tracker = model_info.name
        self.aspect_order = model_info.order
        self.tracked_point_slices = model_info.tracked_point_slices
        self.landmark_slices = model_info.landmark_slices
        self.model_info = model_info
        
        self._initialize_aspects()

    def _initialize_aspects(self):
        """
        Initializes the predefined anatomical aspects (body, face, hands) for the Human instance.
        Aspects are added based on the configuration provided in the ModelInfo instance.
        """
        self._add_body()

        if HumanAspectNames.FACE.value in self.aspect_order:
            self._add_face()

        if HumanAspectNames.LEFT_HAND.value in self.aspect_order:
            self._add_left_hand()
        
        if HumanAspectNames.RIGHT_HAND.value in self.aspect_order:
            self._add_right_hand()

    def _add_body(self):
        body = Aspect.from_model_info(
            name = HumanAspectNames.BODY.value,
            model_info = self.model_info,
            metadata = {"tracker_type": self.tracker}
        )
        self.add_aspect(body)
    
    def _add_face(self):
        face = Aspect.from_model_info(
            name = HumanAspectNames.FACE.value,
            model_info = self.model_info,
            metadata = {"tracker_type": self.tracker}
        )
        self.add_aspect(face)

    def _add_left_hand(self):
        left_hand = Aspect.from_model_info(
            name = HumanAspectNames.LEFT_HAND.value,
            model_info = self.model_info,
            metadata = {"tracker_type": self.tracker}
        )
        self.add_aspect(left_hand)

    def _add_right_hand(self):
        right_hand = Aspect.from_model_info(
            name = HumanAspectNames.RIGHT_HAND.value,
            model_info = self.model_info,
            metadata = {"tracker_type": self.tracker}
        )
        self.add_aspect(right_hand)

    @property
    def body(self) -> Aspect:
        return self.aspects[HumanAspectNames.BODY.value]
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

        self.body.add_tracked_points(tracked_points_numpy_array[:,self.tracked_point_slices[HumanAspectNames.BODY.value],:])

        if HumanAspectNames.FACE.value in self.tracked_point_slices and self.face is not None:
            self.face.add_tracked_points(
                tracked_points_numpy_array[:,self.tracked_point_slices[HumanAspectNames.FACE.value],:]
                )
            
        if HumanAspectNames.LEFT_HAND.value in self.tracked_point_slices and self.left_hand is not None:
            self.left_hand.add_tracked_points(
                tracked_points_numpy_array[:,self.tracked_point_slices[HumanAspectNames.LEFT_HAND.value],:]
                )
        
        if HumanAspectNames.RIGHT_HAND.value in self.tracked_point_slices and self.right_hand is not None:
            self.right_hand.add_tracked_points(
                tracked_points_numpy_array[:,self.tracked_point_slices[HumanAspectNames.RIGHT_HAND.value],:]
                )


    def sort_parquet_dataframe(self, dataframe:pd.DataFrame):
        num_frames = len(list(dataframe['frame'].unique()))
        
        model_names = list(dataframe['model'].unique())

        for name in model_names:
            tracker_name, aspect_name = name.split(".")

            aspect_data_tidy = dataframe[dataframe['model'] == name]
            trajectory_dict = {}
            for trajectory_name in list(aspect_data_tidy['type'].unique()):
                trajectory_data = aspect_data_tidy[aspect_data_tidy['type'] == trajectory_name]
                num_markers = len(list(trajectory_data['keypoint'].unique()))
                trajectory_data_wide = trajectory_data.pivot_table(index="frame", columns="keypoint", values=["x", "y", "z"], dropna = False)
                trajectory_data_wide = trajectory_data_wide.swaplevel(axis=1)
                trajectory_data_wide = trajectory_data_wide.sort_index(axis=1)
                trajectory_array = trajectory_data_wide.to_numpy().reshape(num_frames,num_markers,3)
                trajectory = Trajectory(
                    name = tracker_name,
                    array = trajectory_array,
                    landmark_names=trajectory_data['keypoint'].unique()
                )
                trajectory_dict.update({trajectory_name:trajectory})
                try:
                    HumanAspectNames(aspect_name)
                except ValueError:
                    f"Aspect {aspect_name} not found in expected HumanAspectNames. Skipping."
                    continue
                
            self.aspects.get(aspect_name).add_trajectory(trajectory_dict)

    def add_landmarks_numpy_array(self, landmarks_numpy_array:np.ndarray):
        """
        Takes in landmark data, splits and categorizes it based on the ranges determined by the ModelInfo,
        and adds it as a tracked point Trajectory to the body, and optionally face/hands aspects 
        """
        self.body.add_landmarks(landmarks_numpy_array[:,self.landmark_slices[HumanAspectNames.BODY.value],:])

        if HumanAspectNames.FACE.value in self.landmark_slices and self.face is not None:
            self.face.add_landmarks(
                landmarks_numpy_array[:,self.landmark_slices[HumanAspectNames.FACE.value],:]
                )
        
        if HumanAspectNames.LEFT_HAND.value in self.landmark_slices and self.left_hand is not None:
            self.left_hand.add_landmarks(
                landmarks_numpy_array[:,self.landmark_slices[HumanAspectNames.LEFT_HAND.value],:]
                )

        if HumanAspectNames.RIGHT_HAND.value in self.landmark_slices and self.right_hand is not None:
            self.right_hand.add_landmarks(
                landmarks_numpy_array[:,self.landmark_slices[HumanAspectNames.RIGHT_HAND.value],:]
                )

    def add_reprojection_error_numpy(self, reprojection_error_data: np.ndarray):
        self.body.add_reprojection_error(reprojection_error_data[:, self.tracked_point_slices[HumanAspectNames.BODY.value]])

        if HumanAspectNames.FACE.value in self.tracked_point_slices and self.face is not None:
            self.face.add_reprojection_error(reprojection_error_data[:, self.tracked_point_slices[HumanAspectNames.FACE.value]])
        
        if HumanAspectNames.LEFT_HAND.value in self.tracked_point_slices and self.left_hand is not None:
            self.left_hand.add_reprojection_error(reprojection_error_data[:, self.tracked_point_slices[HumanAspectNames.LEFT_HAND.value]])
        
        if HumanAspectNames.RIGHT_HAND.value in self.tracked_point_slices and self.right_hand is not None:
            self.right_hand.add_reprojection_error(reprojection_error_data[:, self.tracked_point_slices[HumanAspectNames.RIGHT_HAND.value]])
