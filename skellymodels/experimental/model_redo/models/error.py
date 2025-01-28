from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import List


class ErrorValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: np.ndarray
    marker_names: List[str]

    @model_validator(mode="after")
    def validate_data(self):
        if self.data.shape[1] != len(self.marker_names):
            raise ValueError(
                f"Error data must have the same number of markers as input name list. Data has {self.data.shape[1]} markers and list has {len(self.marker_names)} markers.")


class Error:
    """
    A container for 3D motion capture error data.

    The Error class manages time-series error data for a set of markers. It provides methods to access
    the data in various formats (dictionary, numpy array, pandas DataFrame).

    Attributes:
        name (str): Identifier for this trajectory set
        _values (Dict[str, np.ndarray]): Dictionary mapping marker names to their error data
            over time with shape (num_frames, 1)
        _landmark_names (List[str]): Names of the all tracked points (need to firm up some of these naming conventions)
        _num_frames (int): Total number of frames in the trajectory data

    """

    def __init__(self, name: str,
                 data: np.ndarray,
                 marker_names: List[str], ):
        self.name = name
        self._values = {}
        self._landmark_names = marker_names
        self._validate_data(data=data, marker_names=self._landmark_names)
        self._set_error_data(data=data, marker_names=self._landmark_names)
        self._num_frames = data.shape[0]
        self._original_data = data

    def _validate_data(self, data: np.ndarray, marker_names: List[str]):
        ErrorValidator(data=data, marker_names=marker_names)

    def _set_error_data(self, data: np.ndarray, marker_names: List[str]):
        self._values.update({marker_name: data[:, i] for i, marker_name in enumerate(marker_names)})
        self._marker_names = marker_names.copy()

    @property
    def data(self) -> dict:
        return self._values

    @property
    def landmark_data(self):
        return {landmark_name: trajectory for landmark_name, trajectory in self._values.items() if
                landmark_name in self._landmark_names}

    @property
    def landmark_names(self):
        return self._landmark_names

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def as_numpy(self):
        return self._original_data

    @property
    def as_dataframe(self):
        tidy_data = []

        for frame_number in range(self._num_frames):
            frame_data = self.get_frame(frame_number)
            for marker_name, marker_error in frame_data.items():
                tidy_data.append({
                    "frame": frame_number,
                    "keypoint": marker_name,
                    "error": marker_error,
                })

        return pd.DataFrame(tidy_data)

    def get_marker(self, marker_name: str):
        return self._values[marker_name]

    def get_frame(self, frame_number: int):
        return {marker_name: trajectory[frame_number] for marker_name, trajectory in self._values.items()}

    def __str__(self) -> str:

        return f"Error with {self._num_frames} frames and {len(self._marker_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()
