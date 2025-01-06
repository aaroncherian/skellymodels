from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
    
    def __str__(self):
        virtual_markers = (
            f"{len(self.virtual_markers_definitions)} virtual markers"
            if self.virtual_markers_definitions else "No virtual markers"
        )
        segments = (
            f"{len(self.segment_connections)} segments"
            if self.segment_connections else "No segment connections"
        )
        com_definitions = (
            f"{len(self.center_of_mass_definitions)} center of mass definitions"
            if self.center_of_mass_definitions else "No center of mass definitions"
        )
        return (f"  {len(self.landmark_names)} landmarks\n"
                f"  {virtual_markers}\n"
                f"  {segments}\n"
                f"  {com_definitions}")