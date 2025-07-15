from pathlib import Path
import numpy as np
from skellymodels.managers.human import Human
from pprint import pprint
from skellymodels.models.tracking_model_info import MediapipeModelInfo
# import warnings
# warnings.simplefilter('always', DeprecationWarning)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

model_info = MediapipeModelInfo()

recording_path = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")
output_data_path = recording_path/'output_data'
## Choose a path to the directory 
path_to_data = output_data_path/'raw_data'/'mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
data = np.load(path_to_data)

# # Create an Actor
# human:Human = Human(
#             name="human_one", 
#             model_info=model_info
#             )

human = Human.from_tracked_points_numpy_array(name = "human_one ",tracked_points_numpy_array=data, model_info=model_info)

# human:Human = Human.from_data(path_to_data_folder= output_data_path)


# pprint([human.aspects])
# human.fix_hands_to_wrist()
# human.put_skeleton_on_ground()

human.calculate() #does our COM/Rigid bones calculations

human.save_out_numpy_data(output_data_path)
human.save_out_csv_data(output_data_path)
human.save_out_all_data_csv(output_data_path)
human.save_out_all_data_parquet(output_data_path)
human.save_out_all_xyz_numpy_data(output_data_path)

f = 2
# pprint([human.aspects])

