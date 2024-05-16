from skellymodels.create_skeleton import create_skeleton_model


def create_mediapipe_skeleton_model():

    from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo 

    """
    Creates a skeleton model using the mediapipe model
    Returns:
    - An instance of the Skeleton class that represents the complete skeletal model 
    """

    mediapipe_model_info = MediapipeModelInfo()
    skeleton_model = create_skeleton_model(actual_markers=mediapipe_model_info.landmark_names, 
                      num_tracked_points=mediapipe_model_info.num_tracked_points,
                      segment_connections=mediapipe_model_info.segment_connections,
                     virtual_markers=mediapipe_model_info.virtual_markers_definitions,
                     joint_hierarchy= mediapipe_model_info.joint_hierarchy,
                     center_of_mass_info=mediapipe_model_info.center_of_mass_definitions
                )
    
    return skeleton_model