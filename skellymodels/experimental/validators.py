
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, field_validator, model_validator

class LandmarkValidator(BaseModel):
    landmark_names: List[str]

    @field_validator("landmark_names")
    @classmethod
    def validate_landmark_names(cls, landmark_names: List[str]):
        if not isinstance(landmark_names, list):
            raise ValueError("Landmark names must be a list.")
        if not all(isinstance(name, str) for name in landmark_names):
            raise ValueError("All landmark names in list msut be strings.")
        return landmark_names

class VirtualMarkerValidator(BaseModel):
    virtual_markers: Dict[str, Dict[str, List[Union[float, str]]]]
    landmark_names: List[str]

    @model_validator(mode="after")
    def validate_virtual_markers(self):
        # Keep track of all valid marker names (landmarks + defined virtual markers)
        valid_marker_names = set(self.landmark_names)
        
        # We need to validate virtual markers in order, as later ones might depend on earlier ones
        for virtual_marker_name, virtual_marker_values in self.virtual_markers.items():
            marker_names = virtual_marker_values.get("marker_names", [])
            marker_weights = virtual_marker_values.get("marker_weights", [])

            # Basic validation checks
            if len(marker_names) != len(marker_weights):
                raise ValueError(
                    f"The number of marker names must match the number of marker weights for {virtual_marker_name}. "
                    f"Currently there are {len(marker_names)} names and {len(marker_weights)} weights."
                )

            if not isinstance(marker_names, list) or not all(isinstance(name, str) for name in marker_names):
                raise ValueError(f"Marker names must be a list of strings for {marker_names}.")

            if not isinstance(marker_weights, list) or not all(
                isinstance(weight, (int, float)) for weight in marker_weights
            ):
                raise ValueError(f"Marker weights must be a list of numbers for {virtual_marker_name}.")

            # Check if all marker names are in our valid set
            invalid_markers = [name for name in marker_names if name not in valid_marker_names]
            if invalid_markers:
                raise ValueError(
                    f"The following markers used in {virtual_marker_name} are not valid landmarks or previously "
                    f"defined virtual markers: {invalid_markers}"
                )

            # Validate weights sum
            weight_sum = sum(marker_weights)
            if not 0.99 <= weight_sum <= 1.01:  # Allowing a tiny bit of floating-point leniency
                raise ValueError(
                    f"Marker weights must sum to approximately 1 for {virtual_marker_name}. Current sum is {weight_sum}."
                )

            # Add this virtual marker to our valid set for future validations
            valid_marker_names.add(virtual_marker_name)

        return self

class SegmentConnectionsValidator(BaseModel):
    segment_connections: Dict[str, Dict[str, str]]
    marker_names: List[str]

    @model_validator(mode="after")
    def validate_segment_connections(self):
        for segment_name, segment_connection in self.segment_connections.items():
            # Check for required keys
            if "proximal" not in segment_connection or "distal" not in segment_connection:
                raise ValueError(f"Segment connection must have 'proximal' and 'distal' keys for {segment_name}.")

            # Check if proximal and distal markers are strings and exist in marker_names
            if segment_connection["proximal"] not in self.marker_names:
                raise ValueError(
                    f"The proximal marker {segment_connection['proximal']} for {segment_name} is not in the list of markers."
                )
            
            if segment_connection["distal"] not in self.marker_names:
                raise ValueError(
                    f"The distal marker {segment_connection['distal']} for {segment_name} is not in the list of markers."
                )

        return self

class CenterOfMassValidator(BaseModel):
    center_of_mass_definitions: Dict[str, Dict[str, float]]
    segment_connections: Dict[str, Dict[str, str]]

    @model_validator(mode="after")
    def validate_center_of_mass(self):
        for segment_name, com_definition in self.center_of_mass_definitions.items():
            # Check for required keys
            if "segment_com_length" not in com_definition or "segment_com_percentage" not in com_definition:
                raise ValueError(f"Center of mass definition for {segment_name} must have 'segment_com_length' and 'segment_com_percentage' keys for {segment_name}.")

            if segment_name not in self.segment_connections:
                raise ValueError(f"Segment {segment_name} not in segment connections.")

