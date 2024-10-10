from typing import Dict, List, Optional
from skellymodels.model_info.model_info import ModelInfo


class QualisysModelInfo(ModelInfo):
    landmark_names = [
        'head',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hand',
        'right_hand',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        'left_heel',
        'right_heel',
        'left_foot_index',
        'right_foot_index',
        ]
    
    num_tracked_points = len(landmark_names)
    tracked_object_names = ['pose_landmarks']

    virtual_markers_definitions = {
        "neck_center": {
            "marker_names": ["left_shoulder", "right_shoulder"],
            "marker_weights": [0.5, 0.5],
        },
        "hips_center": {
            "marker_names": ["left_hip", "right_hip"],
            "marker_weights": [0.5, 0.5],
        },
    }

    segment_connections = {
            "head": {"proximal": "left_ear", "distal": "right_ear"},
            "spine": {"proximal": "neck_center", "distal": "hips_center"},
            "shoulder": {"proximal": "left_shoulder", "distal": "right_shoulder"},
            "right_upper_arm": {"proximal": "right_shoulder", "distal": "right_elbow"},
            "left_upper_arm": {"proximal": "left_shoulder", "distal": "left_elbow"},
            "right_forearm": {"proximal": "right_elbow", "distal": "right_wrist"},
            "left_forearm": {"proximal": "left_elbow", "distal": "left_wrist"},
            "right_hand": {"proximal": "right_wrist", "distal": "right_hand"},
            "left_hand": {"proximal": "left_wrist", "distal": "left_hand"},
            "hips": {"proximal": "left_hip", "distal": "right_hip"},
            "right_thigh": {"proximal": "right_hip", "distal": "right_knee"},
            "left_thigh": {"proximal": "left_hip", "distal": "left_knee"},
            "right_shank": {"proximal": "right_knee", "distal": "right_ankle"},
            "left_shank": {"proximal": "left_knee", "distal": "left_ankle"},
            "right_foot": {"proximal": "right_ankle", "distal": "right_foot_index"},
            "left_foot": {"proximal": "left_ankle", "distal": "left_foot_index"},
            "right_foot_bottom": {"proximal": "right_heel", "distal": "right_foot_index"},
            "left_foot_bottom": {"proximal": "left_heel", "distal": "left_foot_index"},
            }
    
    center_of_mass_definitions = {
        "head": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.081,
        },
        "spine": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.497,
        },
        "right_upper_arm": {
            "segment_com_length": 0.436,
            "segment_com_percentage": 0.028,
        },
        "left_upper_arm": {
            "segment_com_length": 0.436,
            "segment_com_percentage": 0.028,
        },
        "right_forearm": {
            "segment_com_length": 0.430,
            "segment_com_percentage": 0.016,
        },
        "left_forearm": {
            "segment_com_length": 0.430,
            "segment_com_percentage": 0.016,
        },
        "right_hand": {
            "segment_com_length": 0.506,
            "segment_com_percentage": 0.006,
        },
        "left_hand": {
            "segment_com_length": 0.506,
            "segment_com_percentage": 0.006,
        },
        "right_thigh": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.1,
        },
        "left_thigh": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.1,
        },
        "right_shank": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.0465,
        },
        "left_shank": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.0465,
        },
        "right_foot": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.0145,
        },
        "left_foot": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.0145,
        },
    }


class QualisysMDN_NIHModelInfo(QualisysModelInfo):
    landmark_names = [
    'head',
    'left_ear',
    'right_ear',
    'cspine',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index',
    ] 
    
    num_tracked_points = len(landmark_names)
    tracked_object_names = ['pose_landmarks']


