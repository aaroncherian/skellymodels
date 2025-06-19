from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import datetime
from skellymodels.models.aspect import Aspect
from skellymodels.models.trajectory import Trajectory
from typing import Dict, Optional

from skellymodels.tracker_info.model_info import ModelInfo
from skellymodels.biomechanics.anatomical_calculations import CalculationPipeline, STANDARD_PIPELINE

from pathlib import Path

FREEMOCAP_PARQUET_NAME = 'freemocap_data_by_frame.parquet'

class Actor(ABC):
    """
    The Actor class is a container for multiple Aspects of a single person/creature/object that we track in 3D.
    """

    def __init__(self, name: str, model_info: Optional[
        ModelInfo] = None, **kwargs):
        self.name = name
        self.aspects: Dict[str, Aspect] = {}
        self.tracker = model_info.name #the day we allow for separate pose estimation for different things (body/hands/face), we (I) may regret this
        self.aspect_order = model_info.order
        self.tracked_point_slices = model_info.tracked_point_slices
        self.landmark_slices = model_info.landmark_slices
        self.model_info = model_info

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
    def from_tracked_points_numpy_array(cls, name: str, model_info: ModelInfo, tracked_points_numpy_array: np.ndarray):
        """
        Takes in a numpy array of tracked points and returns an Actor object with the tracked points added in aspects
        """
        actor = cls(name=name, model_info=model_info)
        actor.add_tracked_points_numpy(tracked_points_numpy_array=tracked_points_numpy_array)
        return actor
    
    @classmethod
    def from_data(cls, model_info: ModelInfo, path_to_data_folder: Path|str):
        """
        Given a path to a folder containing the ouput model data, this method will attempt to load the data from a Parquet file.
        """
        #Later on, if needed, we can consider fallback methods to loading from the big CSV or all the individual CSVs as well
        try:
            parquet_file = path_to_data_folder / FREEMOCAP_PARQUET_NAME
            return cls.from_parquet(model_info, parquet_file)
        except FileNotFoundError:
            print(f"Could not find parquet file at {parquet_file}")
        except Exception as e:
            print(f"Failed to load from Parquet: {e}")
            
        raise RuntimeError(f"Could not load data from {path_to_data_folder}")
    
    @classmethod
    def from_parquet(cls, model_info: ModelInfo, path_to_parquet_file: Path|str):
        path_to_parquet_file = Path(path_to_parquet_file)
        dataframe = pd.read_parquet(path_to_parquet_file)

        actor = cls(name =dataframe.attrs['metadata']['name'],
                    model_info = model_info)

        if set(actor.aspect_order) != set(dataframe.attrs['metadata']['aspects']): #Want to come back around and make a more robust check
            raise ValueError(f"Aspects in parquet file {dataframe.attrs['metadata']['aspects']} do not match aspects specified in model info {actor.aspect_order}")

        actor.sort_parquet_dataframe(dataframe)
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

                # Convert trajectory to a DataFrame
                trajectory_df = trajectory.as_dataframe
                
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

    def save_out_numpy_data(self, path_to_output_folder: Optional[Path] = None):
        if path_to_output_folder is None:
            path_to_output_folder = Path.cwd()

        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                save_path = path_to_output_folder / f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.npy"
                np.save(save_path,
                        trajectory.as_array)  # TODO: the .data is throwing a type error because this is sometimes a dict instead of an array
                print(f"Saved out {save_path}")

    def save_out_csv_data(self, path_to_output_folder: Optional[Path] = None):
        if path_to_output_folder is None:
            path_to_output_folder = Path.cwd()

        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                save_path = path_to_output_folder / f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv"
                trajectory.as_dataframe.to_csv(path_to_output_folder/f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv")
                print(f"Saved out {save_path}") 

    def save_out_all_data_csv(self, path_to_output_folder: Optional[Path] = None):
        if path_to_output_folder is None:
            path_to_output_folder = Path.cwd()
        save_path = path_to_output_folder / 'freemocap_data_by_frame.csv'    
        self.all_data_as_dataframe().to_csv(save_path, index=False)
        print(f"Data successfully saved to {save_path}")

    def save_out_all_data_parquet(self, path_to_output_folder: Optional[Path] = None):
        if path_to_output_folder is None:
            path_to_output_folder = Path.cwd()

        dataframe = self.all_data_as_dataframe()
        dataframe.attrs['metadata'] = {
            'created_at': datetime.datetime.now().isoformat(),
            'created_with': 'skelly_models',
            'name': self.name,
            'aspects': list(self.aspect_order)
        }
        save_path = path_to_output_folder / FREEMOCAP_PARQUET_NAME
        dataframe.to_parquet(save_path)
        print(f"Data successfully saved to {save_path}")

    def sort_parquet_dataframe(self, dataframe:pd.DataFrame):
        num_frames = dataframe['frame'].nunique()
        expected_models = {f"{self.tracker}.{aspect_name}" for aspect_name in self.aspects}

        for model_name, aspect_data in dataframe.groupby('model'): #model name is formatted {tracker}.{aspect_name} in our CSV/parquet
            if model_name not in expected_models:
                raise ValueError(f"Aspect {model_name} not found in aspects initialized in Actor: {expected_models}")
            
            tracker_name, aspect_name = model_name.split(".")
            trajectory_dict: dict[str, Trajectory] = {}

            for trajectory_name, trajectory_data in aspect_data.groupby('type'):
                marker_order = trajectory_data["keypoint"].drop_duplicates().tolist()

                num_markers = len(marker_order)

                trajectory_data_wide = (
                    trajectory_data
                    .pivot_table(index="frame", columns="keypoint", values=["x", "y", "z"], dropna = False)
                    .swaplevel(axis=1)
                    .sort_index(axis=1)
                    .reindex(columns = marker_order, level = 0)
                    )
                
                trajectory_array = trajectory_data_wide.to_numpy().reshape(num_frames,num_markers,3)
                
                trajectory = Trajectory(
                    name = trajectory_name,
                    array = trajectory_array,
                    landmark_names = marker_order
                )

                trajectory_dict[trajectory_name] = trajectory
                
            self.aspects.get(aspect_name).add_trajectory(trajectory_dict)