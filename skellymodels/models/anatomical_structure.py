from pydantic import BaseModel
from typing import Dict, List
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection, SegmentCenterOfMassDefinition

class AnatomicalStructure(BaseModel):
    """A data class representing the anatomical structure of a tracked object, defining its
    landmarks, virtual markers, segments, and center of mass definitions and joint hierarchy. 
    The latter three are used in calculating center of mass and enforcing rigid bones for the object.

    The structure itself is built by the AnatomicalStructureBuilder.
    """
    tracked_point_names: List[MarkerName]
    virtual_markers_definitions: Dict[str, VirtualMarkerDefinition]|None = None
    segment_connections: Dict[SegmentName, SegmentConnection]|None = None
    center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]|None = None
    joint_hierarchy: Dict[MarkerName, List[MarkerName]]|None = None

    @property
    def landmark_names(self):
        landmark_names = self.tracked_point_names.copy()
        if self.virtual_markers_definitions:
            landmark_names.extend(self.virtual_markers_definitions.keys())
        return landmark_names

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
        joint_hierarchy = (
            f"{len(self.joint_hierarchy)} joint hierarchies"
            if self.joint_hierarchy else "No joint hierarchy"
        )
        return (f"  {len(self.tracked_point_names)} tracked points\n"
                f"  {virtual_markers}\n"
                f"  {segments}\n"
                f"  {com_definitions}\n"
                f"  {joint_hierarchy}")