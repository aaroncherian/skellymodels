from dataclasses import dataclass
from typing import Any

from skellymodels.model_info.qualisys_model_info import QualisysModelInfo
from pydantic import BaseModel, model_validator, Field, field_validator
from typing import Dict, List, Optional, Union
@dataclass
class AspectInfo:
    name: str
    value: Any

class Aspect:
    def __init__(self, name: str):
        self.name = name
        self.aspect_info = {}

    def add_info(self, aspect_info: AspectInfo):
        self.aspect_info[aspect_info.name] = aspect_info.value

    def __getitem__(self, key: str):
        return self.aspect_info[key]

class Character:
    def __init__(self, name: str):
        self.name = name
        self.aspects = {}

    def add_aspect(self, aspect: Aspect):
        self.aspects[aspect.name] = aspect

    def __getitem__(self, key: str):
        return self.aspects[key]
    
    def __str__(self):
        return self.name


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

    @field_validator("virtual_markers")
    @classmethod
    def validate_virtual_marker(cls, virtual_marker: Dict[str, List[Union[float, str]]]):
        for virtual_marker_name, virtual_marker_values in virtual_marker.items():
            marker_names = virtual_marker_values.get("marker_names", [])
            marker_weights = virtual_marker_values.get("marker_weights", [])

            if len(marker_names) != len(marker_weights):
                raise ValueError(
                    f"The number of marker names must match the number of marker weights for {virtual_marker_name}. Currently there are {len(marker_names)} names and {len(marker_weights)} weights."
                )

            if not isinstance(marker_names, list) or not all(isinstance(name, str) for name in marker_names):
                raise ValueError(f"Marker names must be a list of strings for {marker_names}.")

            if not isinstance(marker_weights, list) or not all(
                isinstance(weight, (int, float)) for weight in marker_weights
            ):
                raise ValueError(f"Marker weights must be a list of numbers for {virtual_marker_name}.")

            weight_sum = sum(marker_weights)
            if not 0.99 <= weight_sum <= 1.01:  # Allowing a tiny bit of floating-point leniency
                raise ValueError(
                    f"Marker weights must sum to approximately 1 for {virtual_marker_name} Current sum is {weight_sum}."
                )
        return virtual_marker

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
    

landmark_names = [
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
    "right_heel",
    "left_heel",
    "right_foot_index",
    "left_foot_index"
    ]

skeleton = Character(name="mediapipe")

body = Aspect(name="body")

skeleton.add_aspect(body)

model_info = QualisysModelInfo()

for key in model_info.to_dict().keys():
    aspect_info = AspectInfo(key, model_info.to_dict()[key])
    body.add_info(aspect_info)

total_marker_names = body["landmark_names"] + list(body["virtual_markers_definitions"].keys()) if body["virtual_markers_definitions"] else body["landmark_names"] 
body.add_info(AspectInfo("marker_names", total_marker_names))


def validate_aspect(aspect: Aspect):
    # Check if virtual_markers_definitions are present
    if aspect["landmark_names"]:
        LandmarkValidator(landmark_names=aspect["landmark_names"])

    if aspect["virtual_markers_definitions"]:
        if not aspect["landmark_names"]:
            raise ValueError("Landmark names must be defined before virtual markers.")
        VirtualMarkerValidator(virtual_markers=aspect["virtual_markers_definitions"])
    
    if aspect["segment_connections"]:
        SegmentConnectionsValidator(segment_connections=aspect["segment_connections"], marker_names=aspect["marker_names"])

    if aspect["center_of_mass_definitions"]:
        if not aspect["segment_connections"]:
            raise ValueError("Segment connections must be defined before center of mass definitions")
        CenterOfMassValidator(center_of_mass_definitions=aspect["center_of_mass_definitions"],
                                segment_connections=aspect["segment_connections"])

        f = 2


    # # If you have other validations:
    # if "segment_connections" in aspect.aspect_info:
    #     SegmentConnectionsInfo(segment_connections=aspect["segment_connections"])
    # if "center_of_mass_definitions" in aspect.aspect_info:
    #     CenterOfMassInfo(com_definitions=aspect["center_of_mass_definitions"])


validate_aspect(body)
f = 2
marker_names = body["landmark_names"]

if "virtual_markers_definitions" in body.aspect_info:
    virtual_markers = body["virtual_markers_definitions"]

    for virtual_marker_name, virtual_marker_values in virtual_markers.items():
        all_marker_names = virtual_marker_values["marker_names"]

body.add_info(AspectInfo("marker_names", all_marker_names))
f = 2
