from skellymodels.models.anatomical_structure import AnatomicalStructure
from skellymodels.tracker_info.model_info import ModelInfo

from pydantic import BaseModel, field_validator, model_validator
from typing import  List, Dict
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection, SegmentCenterOfMassDefinition

class AnatomicalStructureBuilder:
    """
        A builder class that constructs AnatomicalStructure instances from data defined in the model info. 

        This builder ensures that anatomical structures are created with proper validation and
        dependency checking between components (e.g., virtual markers require tracked points to exist).

    """
    def __init__(self):
        self.tracked_point_names: List[MarkerName]|None = None
        self.virtual_markers_definitions: Dict[str, VirtualMarkerDefinition]|None = None
        self.segment_connections: Dict[SegmentName, SegmentConnection]|None = None 
        self.center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]|None = None
        self.joint_hierarchy: Dict[MarkerName, List[MarkerName]]|None = None

    @property
    def _marker_names(self):
        if not self.tracked_point_names:
            raise ValueError("Tracked point names must be set before calling for a marker list.")
        markers = self.tracked_point_names.copy()
        if self.virtual_markers_definitions:
            markers.extend(self.virtual_markers_definitions.keys())
        return markers

    def with_tracked_points(self, tracked_point_names: List[MarkerName]):
        TrackedPointsValidator(tracked_point_names=tracked_point_names)
        self.tracked_point_names = tracked_point_names.copy()
        return self

    def with_virtual_markers(self, virtual_marker_definitions: Dict[str, VirtualMarkerDefinition]|None):
        
        if virtual_marker_definitions is not None:
            if not self.tracked_point_names:
                raise ValueError("Tracked point names must be set before adding virtual markers.")
            
            VirtualMarkerValidator(virtual_markers=virtual_marker_definitions,
                                tracked_point_names=self.tracked_point_names)
        self.virtual_markers_definitions = virtual_marker_definitions
        return self

    def with_segment_connections(self, segment_connections: Dict[SegmentName, SegmentConnection]|None):

        if segment_connections is not None:
            SegmentConnectionsValidator(segment_connections=segment_connections,
                                        marker_names=self._marker_names)
                                        
        self.segment_connections = segment_connections
        return self

    def with_center_of_mass(self, center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]|None):
        
        if center_of_mass_definitions is not None:
            if not self.segment_connections:
                raise ValueError("Segment connections must be set before adding center of mass definitions")
            
            CenterOfMassValidator(center_of_mass_definitions=center_of_mass_definitions,
                                    segment_connections=self.segment_connections)
        self.center_of_mass_definitions = center_of_mass_definitions
        return self
    
    def with_joint_hierarchy(self,joint_hierarchy: Dict[MarkerName, List[MarkerName]]|None):
        if joint_hierarchy is not None:
            JointHierarchyValidator(joint_hierarchy=joint_hierarchy,
                                    marker_names=self._marker_names)
            
        self.joint_hierarchy = joint_hierarchy
        return self

    def build(self):
        if not self.tracked_point_names:
            raise ValueError("Cannot build AnatomicalStructure without tracked point names")
        return AnatomicalStructure(tracked_point_names=self.tracked_point_names,
                                   virtual_markers_definitions=self.virtual_markers_definitions,
                                   segment_connections=self.segment_connections,
                                   center_of_mass_definitions=self.center_of_mass_definitions,
                                   joint_hierarchy=self.joint_hierarchy)


def create_anatomical_structure_from_model_info(model_info:ModelInfo) -> Dict[str, AnatomicalStructure]:
        structures = {}
        for aspect_name, aspect_structure in model_info.aspects.items():

            structures[aspect_name] = (AnatomicalStructureBuilder()
                .with_tracked_points(aspect_structure.tracked_points_names)
                .with_virtual_markers(aspect_structure.virtual_marker_definitions)
                .with_segment_connections(aspect_structure.segment_connections)
                .with_center_of_mass(aspect_structure.center_of_mass_definitions)
                .with_joint_hierarchy(aspect_structure.joint_hierarchy)
            ).build()

        return structures

class TrackedPointsValidator(BaseModel):
    tracked_point_names: List[MarkerName]

class VirtualMarkerValidator(BaseModel):
    virtual_markers: Dict[str, VirtualMarkerDefinition]
    tracked_point_names: List[MarkerName]

    @model_validator(mode="after")
    def validate_virtual_markers(self):
        # Keep track of all valid marker names (tracked points + defined virtual markers)
        valid_marker_names = set(self.tracked_point_names)
        
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


            # Check if all marker names are in our valid set
            invalid_markers = [name for name in marker_names if name not in valid_marker_names]
            if invalid_markers:
                raise ValueError(
                    f"The following markers referenced in {virtual_marker_name} are not valid tracked points"
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
    segment_connections: Dict[SegmentName, SegmentConnection]
    marker_names: List[MarkerName]

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
    center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]
    segment_connections: Dict[SegmentName, SegmentConnection]

    @model_validator(mode="after")
    def validate_center_of_mass(self):
        for segment_name, com_definition in self.center_of_mass_definitions.items():
            # Check for required keys
            if "segment_com_length" not in com_definition or "segment_com_percentage" not in com_definition:
                raise ValueError(f"Center of mass definition for {segment_name} must have 'segment_com_length' and 'segment_com_percentage' keys for {segment_name}.")

            if segment_name not in self.segment_connections:
                raise ValueError(f"Segment {segment_name} not in segment connections.")


class JointHierarchyValidator(BaseModel):
    joint_hierarchy: Dict[MarkerName, List[MarkerName]]
    marker_names: List[MarkerName]

    @model_validator(mode="after")
    def validate_joint_hierarchy(self):
        for joint_name, joint_hierarchy in self.joint_hierarchy.items():

            if joint_name not in self.marker_names:
                raise ValueError(f"Joint name {joint_name} not in list of markers.")

            if not all(marker in self.marker_names for marker in joint_hierarchy):
                raise ValueError(f"Joint hierarchy for {joint_name} contains markers not in the list of markers.")
        return self