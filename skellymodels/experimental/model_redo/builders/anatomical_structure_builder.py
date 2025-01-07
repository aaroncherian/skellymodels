from skellymodels.experimental.model_redo.models.anatomical_structure import AnatomicalStructure
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List, Dict, Union

class AnatomicalStructureBuilder:
    def __init__(self):
        self.landmark_names: Optional[List[str]] = None
        self.virtual_markers_definitions: Optional[Dict[str, Dict[str, List[Union[float, str]]]]] = None
        self.segment_connections: Optional[Dict[str, Dict[str, str]]] = None
        self.center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None
        self.joint_hierarchy: Optional[Dict[str, List[str]]] = None

    @property
    def _marker_names(self):
        if not self.landmark_names:
            raise ValueError("Landmark names must be set before calling for a marker list.")
        markers = self.landmark_names.copy()
        if self.virtual_markers_definitions:
            markers.extend(self.virtual_markers_definitions.keys())
        return markers

    def with_landmarks(self, landmark_names: List[str]):
        LandmarkValidator(landmark_names=landmark_names)
        self.landmark_names = landmark_names.copy()
        return self

    def with_virtual_markers(self, virtual_marker_definitions: Dict[str, Dict[str, List[Union[float, str]]]]):
        if not self.landmark_names:
            raise ValueError("Landmark names must be set before adding virtual markers.")
        VirtualMarkerValidator(virtual_markers=virtual_marker_definitions,
                               landmark_names=self.landmark_names)
        self.virtual_markers_definitions = virtual_marker_definitions
        return self

    def with_segment_connections(self, segment_connections: Dict[str, Dict[str, str]]):
        SegmentConnectionsValidator(segment_connections=segment_connections,
                                    marker_names=self._marker_names)
        self.segment_connections = segment_connections
        return self

    def with_center_of_mass(self, center_of_mass_definitions: Dict[str, Dict[str, float]]):
        if not self.segment_connections:
            raise ValueError("Segment connections must be set before adding center of mass definitions")
        CenterOfMassValidator(center_of_mass_definitions=center_of_mass_definitions,
                                segment_connections=self.segment_connections)
        self.center_of_mass_definitions = center_of_mass_definitions
        return self
    
    def with_joint_hierarchy(self,joint_hierarchy: Dict[str, List[str]]):
        JointHierarchyValidator(joint_hierarchy=joint_hierarchy,
                                marker_names=self._marker_names)
        self.joint_hierarchy = joint_hierarchy
        return self

    def build(self):
        if not self.landmark_names:
            raise ValueError("Cannot build AnatomicalStructure without landmark names")
        return AnatomicalStructure(landmark_names=self.landmark_names,
                                   virtual_markers_definitions=self.virtual_markers_definitions,
                                   segment_connections=self.segment_connections,
                                   center_of_mass_definitions=self.center_of_mass_definitions,
                                   joint_hierarchy=self.joint_hierarchy)




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


class JointHierarchyValidator(BaseModel):
    joint_hierarchy: Dict[str, List[str]]
    marker_names: List[str]

    @model_validator(mode="after")
    def validate_joint_hierarchy(self):
        for joint_name, joint_hierarchy in self.joint_hierarchy.items():

            if joint_name not in self.marker_names:
                raise ValueError(f"Joint name {joint_name} not in list of markers.")

            if not all(marker in self.marker_names for marker in joint_hierarchy):
                raise ValueError(f"Joint hierarchy for {joint_name} contains markers not in the list of markers.")
        return self