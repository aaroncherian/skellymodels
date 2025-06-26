from pathlib import Path
import numpy as np
from skellymodels.managers.human import Human
from pprint import pprint
from skellymodels.models.tracking_model_info import MediapipeModelInfo


model_info = MediapipeModelInfo()

## Choose a path to the directory 
path_to_data = Path(r"C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_test_data\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")
data = np.load(path_to_data)

## Create an Actor
human = Human(
            name="human_one", 
            model_info=model_info
            )

human.add_tracked_points_numpy(tracked_points_numpy_array=data)
pprint([human.aspects])

human.calculate() #does our COM/Rigid bones calculations
    
# human.save_out_numpy_data()
# human.save_out_csv_data()
# human.save_out_all_data_csv()
# human.save_out_all_data_parquet()

f = 2
# pprint([human.aspects])

