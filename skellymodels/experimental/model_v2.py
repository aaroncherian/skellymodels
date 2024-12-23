from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from skellymodels.experimental.validators import (LandmarkValidator, 
                                                  VirtualMarkerValidator, 
                                                  SegmentConnectionsValidator,
                                                  CenterOfMassValidator)
from skellymodels.model_info.qualisys_model_info import QualisysModelInfo

class Character:
    def __init__(self, name: str):
        self.name = name
        self.aspects = {}

    def __getitem__(self, key: str):
        return self.aspects[key]
    
    def __str__(self):
        return self.name
    

class Aspect:
    def __init__(self, name:str):
        self.name = name
        self.structure = {}
        self.data = {}
        self.metadata = {}

    def add_structure(self, structure: Any):
        self.structure = structure
        

@dataclass
class AnatomicalStructure:
    landmark_names: List[str]
    virtual_markers_definitions: Optional[Dict[str, Dict[str, List[Union[float, str]]]]] = None
    segment_connections: Optional[Dict[str, Dict[str, str]]] = None
    center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def marker_names(self):
        markers = self.landmark_names.copy()
        if self.virtual_markers_definitions:
            markers.extend(self.virtual_markers_definitions.keys())
        return markers
    
    @property
    def virtual_marker_names(self):
        if not self.virtual_markers_definitions:
            return []
        return list(self.virtual_markers_definitions.keys())



class AnatomicalStructureBuilder:
    def __init__(self):
        self.landmark_names: Optional[List[str]] = None
        self.virtual_markers_definitions: Optional[Dict[str, Dict[str, List[Union[float, str]]]]] = None
        self.segment_connections: Optional[Dict[str, Dict[str, str]]] = None
        self.center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None

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

    def build(self):
        if not self.landmark_names:
            raise ValueError("Cannot build AnatomicalStructure without landmark names")
        return AnatomicalStructure(landmark_names=self.landmark_names,
                                   virtual_markers_definitions=self.virtual_markers_definitions,
                                   segment_connections=self.segment_connections,
                                   center_of_mass_definitions=self.center_of_mass_definitions)



skeleton = Character(name="mediapipe")

structure = (AnatomicalStructureBuilder()
             .with_landmarks(QualisysModelInfo().landmark_names)
             .with_virtual_markers(QualisysModelInfo().virtual_markers_definitions)
             .with_segment_connections(QualisysModelInfo().segment_connections)
             .with_center_of_mass(QualisysModelInfo().center_of_mass_definitions)
             .build()
)

aspect = Aspect(name="body")
aspect.add_structure(structure)

f = 2,
