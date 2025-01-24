from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from skellymodels.experimental.model_redo.models.aspect import Aspect
from typing import Dict, Optional

from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo

class Actor(ABC):
    """
    The Actor class is a container for multiple Aspects of a single person/creature/object that we track in 3D.
    """
    def __init__(self, name: str, model_info: Optional[ModelInfo] = None): # depending on what we're using model info for, this could maybe just be **kwargs
        self.name = name
        self.actor_type = "generic"
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
    
    @abstractmethod
    def add_tracked_points_numpy(self, tracked_points_numpy_array: np.ndarray):
        """
        Takes in the tracked points array, splits and categorizes it based on the ranges determined by the ModelInfo,
        and adds it as a tracked point Trajectory to the body, and optionally face/hands aspects 
        """
        pass

    @classmethod
    def from_numpy_array(cls, name:str, model_info: ModelInfo, tracked_points_numpy_array:np.ndarray):
        """
        Takes in a numpy array of tracked points and returns an Actor object with the tracked points added in aspects
        """
        actor = cls(name=name, model_info=model_info)
        actor.add_tracked_points_numpy(tracked_points_numpy_array=tracked_points_numpy_array)
        return actor
    
    def save_out_numpy_data(self):
        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                print('Saving out numpy:', aspect.metadata['tracker_type'], aspect.name, trajectory.name)
                np.save(f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.npy", trajectory.data)  # TODO: the .data is hrowing a type error because this is sometimes a dict instead of an array

    def save_out_csv_data(self):
        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                print('Saving out CSV:', aspect.metadata['tracker_type'], aspect.name, trajectory.name)
                trajectory.as_dataframe.to_csv(f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv")

    def save_out_all_data_csv(self):
        all_data = []

        # Loop through aspects
        for aspect_name, aspect in self.aspects.items():
            if '3d_xyz' in aspect.trajectories:
                # Get tidy DataFrame for the trajectory
                trajectory_df = aspect.trajectories['3d_xyz'].as_dataframe
                
                # Add metadata column for model
                trajectory_df['model'] = f"{aspect.metadata['tracker_type']}_{aspect_name}"
                
                # Append DataFrame to the list
                all_data.append(trajectory_df)

        # Combine all DataFrames into one
        big_df = pd.concat(all_data, ignore_index=True)

        # Sort by frame and then by model
        big_df = big_df.sort_values(by=['frame', 'model']).reset_index(drop=True)

        # Save the result to CSV
        big_df.to_csv('freemocap_data_by_frame.csv', index=False)
        print("Data successfully saved to 'freemocap_data_by_frame.csv'")


        