from typing import Dict, List, Optional


class ModelInfo:
    landmark_names: List[str]
    num_tracked_points: int
    tracked_object_names: list
    virtual_markers_definitions: Optional[Dict[str, Dict[str, List[str | float]]]] = None
    segment_connections: Optional[Dict[str, Dict[str, str]]] = None
    center_of_mass_definitions: Optional[Dict[str, Dict[str, float]]] = None
    joint_hierarchy: Optional[Dict[str, List[str]]] = None

    @classmethod
    def to_dict(cls) -> Dict[str, any]:
        return {
            "landmark_names": cls.landmark_names,
            "num_tracked_points": cls.num_tracked_points,
            "tracked_object_names": cls.tracked_object_names,
            "virtual_markers_definitions": cls.virtual_markers_definitions,
            "segment_connections": cls.segment_connections,
            "center_of_mass_definitions": cls.center_of_mass_definitions,
            "joint_hierarchy": cls.joint_hierarchy,
        }