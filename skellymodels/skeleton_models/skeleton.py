import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict
import numpy as np
from skellymodels.skeleton_models.marker_info import MarkerInfo
from skellymodels.skeleton_models.segments import Segment, Segments, SegmentAnthropometry
import json


class Skeleton(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    markers: MarkerInfo
    num_tracked_points: int
    segments: Optional[Dict[str, Segment]] = None
    # original_marker_data: Dict[str, np.ndarray] = {}
    # virtual_marker_data: Dict[str, np.ndarray] = {}
    _marker_data: Dict[str, np.ndarray] = {}
    rigid_marker_data: Dict[str, np.ndarray] = {}
    joint_hierarchy: Optional[Dict[str, List[str]]] = None
    center_of_mass_definitions: Optional[Dict[str, SegmentAnthropometry]] = None
    num_frames: Optional[int] = None

    def add_segments(self, segment_connections: Dict[str, Segment]) -> None:
        """
        Adds segment connection data to the skeleton model.

        Parameters:
        - segment_connections: A dictionary where each key is a segment name and its value is a dictionary
          with information about that segment (e.g., 'proximal', 'distal' marker names).
        """
        segments_model = Segments(
            markers=self.markers,
            segment_connections={name: segment for name, segment in segment_connections.items()},
        )
        self.segments = segments_model.segment_connections

    def add_joint_hierarchy(self, joint_hierarchy: Dict[str, List[str]]) -> None:
        """
        Adds joint hierarchy data to the skeleton model.

        Parameters:
        - joint_hierarchy: A dictionary with joint names as keys and lists of connected marker names as values.
        """
        for joint_name, joint_connections in joint_hierarchy.items():
            if joint_name not in self.markers.all_markers:
                raise ValueError(f"The joint {joint_name} is not in the list of markers or virtual markers.")
            for connected_marker in joint_connections:
                if connected_marker not in self.markers.all_markers:
                    raise ValueError(
                        f"The connected marker {connected_marker} for {joint_name} is not in the list of markers or virtual markers."
                    )
        self.joint_hierarchy = joint_hierarchy

    def add_center_of_mass_definitions(self, center_of_mass_definitions: Dict[str, Dict[str, float]]) -> None:
        """
        Adds anthropometric center of mass definitions to the skeleton model.

        Parameters:
        - center_of_mass_definitions: A dictionary containing segment mass percentages.
        """
        if self.segments is None:
            raise ValueError("Segments must be defined before center of mass definitions can be added.")
        for com_segment_name in center_of_mass_definitions.keys():
            if com_segment_name not in self.segments.keys():
                raise ValueError(f"Segment {com_segment_name} is not in the list of segments.")
        self.center_of_mass_definitions = {
            name: SegmentAnthropometry(**values) for name, values in center_of_mass_definitions.items()
        }

    def integrate_freemocap_3d_data(self, freemocap_3d_data: np.ndarray) -> None:
        """
        Integrates 3D data from FreeMoCap into the skeleton model.

        Parameters:
        - freemocap_3d_data: NumPy array with dimensions (num_frames, num_markers, 3).
        """
        self.num_frames = freemocap_3d_data.shape[0]
        num_markers_in_data = freemocap_3d_data.shape[1]
        original_marker_names_list = self.original_marker_names

        if num_markers_in_data != self.num_tracked_points:
            raise ValueError(
                f"The number of markers in the 3D data ({num_markers_in_data}) does not match "
                f"the expected number of tracked points ({self.num_tracked_points})."
            )

        self._marker_data = {
            marker_name: freemocap_3d_data[:, i, :] for i, marker_name in enumerate(original_marker_names_list)
        }

        try:
            self.calculate_virtual_markers()
        except ValueError:
            print(
                "Freemocap data integrated without virtual markers, as no virtual marker definition was provided"
            )

    def integrate_rigid_marker_data(self, rigid_marker_data: np.ndarray) -> None:
        """
        Integrates rigid body data into the skeleton model.vc

        Parameters:
        - rigid_marker_data: NumPy array with dimensions (num_frames, num_markers, 3).
        """
        self.num_frames = rigid_marker_data.shape[0]
        num_markers_in_data = rigid_marker_data.shape[1]
      

        if num_markers_in_data != len(self.marker_names):
            raise ValueError(
                f"The number of markers in the 3D data ({num_markers_in_data}) does not match "
                f"the expected number of markers ({len(self.marker_names)})."
            )

        self.rigid_marker_data = {
            marker_name: rigid_marker_data[:, i, :] for i, marker_name in enumerate(self.marker_names)
        }

        self._marker_data = self.rigid_marker_data


    def calculate_virtual_markers(self) -> None:
        """
        Calculates the positions of virtual markers based on the original marker data.
        """
        if not self._marker_data:
            raise ValueError(
                "3D marker data must be integrated before calculating virtual markers. Run `integrate_freemocap_3d_data()` first."
            )
        if not self.markers.virtual_marker_definition:
            raise ValueError(
                "Virtual marker info must be defined before calculating virtual markers. Run `add_virtual_markers()` first."
            )
        
        virtual_marker_data = {}
        for vm_name, vm_info in self.markers.virtual_marker_definition.virtual_markers.items():
            vm_positions = np.zeros((self.num_frames, 3))
            for marker_name, weight in zip(vm_info["marker_names"], vm_info["marker_weights"]):
                vm_positions += self._marker_data[marker_name] * weight
            virtual_marker_data[vm_name] = vm_positions

        self._marker_data.update(virtual_marker_data)

    def get_segment_markers(self, segment_name: str) -> Dict[str, np.ndarray]:
        """Returns a dictionary with the positions of the proximal and distal markers for a segment."""
        if not self.segments:
            raise ValueError("Segments must be defined before getting segment markers.")
        if not self.trajectories:
            raise ValueError("Trajectories must be defined before getting segment markers.")
        segment = self.segments.get(segment_name)
        if not segment:
            raise ValueError(f"Segment '{segment_name}' is not defined in the skeleton.")
        proximal_trajectories = self.trajectories.get(segment.proximal)
        distal_trajectories = self.trajectories.get(segment.distal)
        return {"proximal": proximal_trajectories, "distal": distal_trajectories}

    @property
    def marker_data_as_numpy(self) -> np.ndarray:
        """
        Converts the marker data dictionary to a NumPy array in (frame, marker, dimension) format.

        Returns:
        - A NumPy array with dimensions (num_frames, num_markers, 3).
        """
        marker_names = self.marker_names
        num_frames = self.num_frames
        num_markers = len(marker_names)
        data_array = np.zeros((num_frames, num_markers, 3))

        for i, marker_name in enumerate(marker_names):
            data_array[:, i, :] = self._marker_data[marker_name]
        return data_array

    
    @property
    def original_marker_data_as_numpy(self) -> np.ndarray:
        """
        Converts the original marker data dictionary to a NumPy array in (frame, marker, dimension) format.

        Returns:
        - A NumPy array with dimensions (num_frames, num_markers, 3).
        """
        marker_names = self.original_marker_names
        num_frames = self.num_frames
        num_markers = len(marker_names)
        data_array = np.zeros((num_frames, num_markers, 3))
        for i, marker_name in enumerate(marker_names):
            data_array[:, i, :] = self._marker_data[marker_name]
        return data_array

    @property
    def trajectories(self) -> Dict[str, np.ndarray]:
        """
        Returns the marker data dictionary.

        Returns:
        - A dictionary of all marker names (original and virtual, if included)
        """
        return self._marker_data
    
    @property
    def marker_names(self) -> List[str]:
        return self.markers.all_markers
    
    @property
    def original_marker_names(self) -> List[str]:
        return self.markers.original_marker_names
    
    @property
    def virtual_marker_names(self) -> List[str]:
        try:
            return list(self.markers.virtual_marker_definition.virtual_markers.keys())
        except AttributeError:
            print('Virtual marker names are not available. No virtual markers are defined.')
            return []
        

    def to_custom_dict(self) -> Dict[str, Any]:
        """
        Converts the Skeleton instance to a dictionary with only the necessary properties for visualization.
        """
        def numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_list(i) for i in obj]
            return obj

        custom_dict = {
            'markers': self.marker_names,
            'trajectories': numpy_to_list(self.trajectories),
            'segments': {k: v.__dict__ for k, v in self.segments.items()} if self.segments else None,
            'num_frames': self.num_frames
        }
        return custom_dict
    
    def to_json(self) -> str:
        """
        Serializes the Skeleton model to a custom JSON string with only the necessary properties.
        """
        custom_dict = self.to_custom_dict()
        return json.dumps(custom_dict)
