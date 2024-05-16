from skellymodels.create_skeleton import create_skeleton_model
from skellymodels.model_info.model_info import ModelInfo


def create_mediapipe_skeleton_model():

        from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo 

        """
        Creates a skeleton model using the mediapipe model
        Returns:
        - An instance of the Skeleton class that represents the complete skeletal model 
        """

        skeleton_model = create_skeleton_from_this_model_info(
            model_info=MediapipeModelInfo()
        )

        return skeleton_model


def create_qualisys_skeleton_model():
    
        from skellymodels.model_info.qualisys_model_info import QualisysModelInfo
    
        """
        Creates a skeleton model using the qualisys model
        Returns:
        - An instance of the Skeleton class that represents the complete skeletal model 
        """

        skeleton_model = create_skeleton_from_this_model_info(
            model_info=QualisysModelInfo()
        )
        
        return skeleton_model


def create_qualisys_mdn_nih_skeleton_model():
      
      from skellymodels.model_info.qualisys_model_info import QualisysMDN_NIHModelInfo

      skeleton_model = create_skeleton_from_this_model_info(
            model_info=QualisysMDN_NIHModelInfo()
      )

      return skeleton_model



def create_skeleton_from_this_model_info(model_info:ModelInfo):
      
        """
        Creates a skeleton model using the qualisys model
        Returns:
        - An instance of the Skeleton class that represents the complete skeletal model 
        """

        skeleton_model = create_skeleton_model(actual_markers=model_info.landmark_names, 
                        num_tracked_points=model_info.num_tracked_points,
                        segment_connections=model_info.segment_connections,
                        virtual_markers=model_info.virtual_markers_definitions,
                        joint_hierarchy= model_info.joint_hierarchy,
                        center_of_mass_info=model_info.center_of_mass_definitions
                    )   
        
        return skeleton_model