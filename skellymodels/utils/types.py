from typing import TypedDict, TypeAlias, List, Dict

MarkerName: TypeAlias = str
SegmentName: TypeAlias = str

class VirtualMarkerDefinition(TypedDict):
    marker_names: List[MarkerName]
    marker_weights: list[float]

class SegmentConnection(TypedDict):
    proximal: MarkerName
    distal: MarkerName

class SegmentCenterOfMassDefinition(TypedDict):
    segment_com_length : float
    segment_com_percentage: float

JointHierarchy: TypeAlias = Dict[MarkerName, List[MarkerName]]