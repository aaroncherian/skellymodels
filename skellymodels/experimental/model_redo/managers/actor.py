from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import datetime
from skellymodels.experimental.model_redo.models.aspect import Aspect
from typing import Dict, Optional

from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from skellymodels.experimental.model_redo.biomechanics.anatomical_calculations import CalculationPipeline, STANDARD_PIPELINE


class Actor(ABC):
    """
    The Actor class is a container for multiple Aspects of a single person/creature/object that we track in 3D.
    """

    def __init__(self, name: str, model_info: Optional[
        ModelInfo] = None, **kwargs):
        self.name = name
        self.aspects: Dict[str, Aspect] = {}

    def __getitem__(self, key: str):
        return self.aspects[key]

    def __str__(self):
        return str(self.aspects.keys())

    def add_aspect(self, aspect: Aspect):
        self.aspects[aspect.name] = aspect

    def get_data(self, aspect_name: str, type: str):
        return self.aspects[aspect_name].trajectories[type].data

    def get_marker_data(self, aspect_name: str, type: str, marker_name: str):
        return self.aspects[aspect_name].trajectories[type].get_marker(marker_name)

    def get_frame(self, aspect_name: str, type: str, frame_number: int):
        return self.aspects[aspect_name].trajectories[type].get_frame(frame_number)

    def get_error_marker(self, aspect_name: str, marker_name: str):
        return self.aspects[aspect_name].reprojection_error.get_marker(marker_name) if self.aspects[
            aspect_name].reprojection_error else None

    def get_error_frame(self, aspect_name: str, frame_number: int):
        return self.aspects[aspect_name].reprojection_error.get_frame(frame_number) if self.aspects[
            aspect_name].reprojection_error else None

    @abstractmethod
    def add_tracked_points_numpy(self, tracked_points_numpy_array: np.ndarray):
        """
        Takes in the tracked points array, splits and categorizes it based on the ranges determined by the ModelInfo,
        and adds it as a tracked point Trajectory to the body, and optionally face/hands aspects 
        """
        pass

    @classmethod
    def from_numpy_array(cls, name: str, model_info: ModelInfo, tracked_points_numpy_array: np.ndarray):
        """
        Takes in a numpy array of tracked points and returns an Actor object with the tracked points added in aspects
        """
        actor = cls(name=name, model_info=model_info)
        actor.add_tracked_points_numpy(tracked_points_numpy_array=tracked_points_numpy_array)
        return actor
    
    def calculate(self, pipeline:CalculationPipeline = STANDARD_PIPELINE):
        for aspect in self.aspects.values():
            results_logs = pipeline.run(aspect=aspect)

            print(f"\nResults for aspect {aspect.name}:")
            for msg in results_logs:
                print(f"  {msg}")



    def all_data_as_dataframe(self) -> pd.DataFrame:
        all_data = []

        # Loop through aspects
        for aspect_name, aspect in self.aspects.items():
            for trajectory_name, trajectory in aspect.trajectories.items():

                if not trajectory_name == 'rigid_3d_xyz': #NOTE: 03/04/25 - excluding rigid body trajectories for the moment because I think they'll be handled different enough in the final version that its not worth including at the moment
                     # Convert trajectory to a DataFrame
                    trajectory_df = trajectory.as_dataframe.copy()
                    
                    # Add metadata columns
                    trajectory_df['model'] = f"{aspect.metadata['tracker_type']}.{aspect_name}"
                    trajectory_df['type'] = trajectory_name  # Store the trajectory type

                    # Add error column
                    if aspect.reprojection_error is None:
                        trajectory_df['reprojection_error'] = np.nan
                    else:
                        trajectory_df['reprojection_error'] = trajectory_df.apply(
                            lambda row: aspect.reprojection_error.get_frame(frame_number=row['frame']).get(row['keypoint'], np.nan),
                            axis=1
                        )  

                    # Append DataFrame to the list
                    all_data.append(trajectory_df)

        # Combine all DataFrames into one
        big_df = pd.concat(all_data, ignore_index=True)

        # Sort by frame, model, and type
        big_df = big_df.sort_values(by=['frame', 'model', 'type']).reset_index(drop=True)

        return big_df


    def save_out_numpy_data(self):
        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                print('Saving out numpy:', aspect.metadata['tracker_type'], aspect.name, trajectory.name)
                np.save(f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.npy",
                        trajectory.data)  # TODO: the .data is throwing a type error because this is sometimes a dict instead of an array

    def save_out_csv_data(self):
        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                print('Saving out CSV:', aspect.metadata['tracker_type'], aspect.name, trajectory.name)
                trajectory.as_dataframe.to_csv(f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv")

    def save_out_all_data_csv(self):
        self.all_data_as_dataframe().to_csv('freemocap_data_by_frame.csv', index=False)
        print("Data successfully saved to 'freemocap_data_by_frame.csv'")

    def save_out_all_data_parquet(self):
        dataframe = self.all_data_as_dataframe()

        dataframe.attrs['metadata'] = {
            'created_at': datetime.datetime.now().isoformat(),
            'created_with': 'skelly_models'
        }
        dataframe.to_parquet('freemocap_data_by_frame.parquet')
        print("Data successfully saved to 'freemocap_data_by_frame.parquet'")
