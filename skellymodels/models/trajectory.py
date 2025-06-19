from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import Dict, List
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection
import warnings
class Trajectory(BaseModel):
    name: str
    array: np.ndarray
    landmark_names: List[MarkerName]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check_shape(self):
        if self.array.shape[1] != len(self.landmark_names):
            raise ValueError(
                f"{self.name}: data has {self.array.shape[1]} columns but "
                f"{len(self.landmark_names)} marker names supplied"
            )
        if self.array.shape[2] != 3:
            raise ValueError(f"{self.name}: last dim must be 3 (xyz)")
        return self

    @property
    def as_array(self) -> np.ndarray:
        return self.array
    
    @property
    def as_dict(self) -> dict[MarkerName, np.ndarray]:
        return {n: self.array[:, i, :]
            for i, n in enumerate(self.landmark_names)}

    @property
    def as_dataframe(self) -> pd.DataFrame:

        df = pd.DataFrame(
            self.array.reshape(self.num_frames*self.num_markers,3),
            columns=['x','y','z'],
        )
        
        df['frame'] = np.repeat(np.arange(self.num_frames),self.num_markers)
        df['keypoint'] = np.tile(self.landmark_names, self.num_frames)
        return df[['frame', 'keypoint', 'x', 'y', 'z']]
         
    @property
    def num_frames(self) -> int:
        return self.array.shape[0]
    
    @property
    def num_markers(self) -> int:
        return self.array.shape[1]
    
    def segment_data(self, segment_connections:Dict[SegmentName, SegmentConnection]) -> Dict[SegmentName, Dict[str, np.ndarray]]:
        if not segment_connections:
            return {}
        d = self.as_dict
        segment_positions = {}
        for name, connection in segment_connections.items():
            proximal = d.get(connection["proximal"])
            distal = d.get(connection["distal"])

            segment_positions.update({name: {'proximal': proximal, 'distal': distal}})

        return segment_positions
    
    @property
    def data(self):
        warnings.warn(".data is deprecated - use .as_dict for the same output",
                      DeprecationWarning,
                      stacklevel=2)  # TODO: elsewhere this is used for the numpy array, but here its a dictionary 
        return self.as_dict

    
    def __str__(self) -> str:
        return f"Trajectory with {self.num_frames} frames and {len(self.landmark_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()
