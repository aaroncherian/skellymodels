from pathlib import Path
import numpy as np

from skellymodels.experimental.model_redo.managers.human import Human

from pprint import pprint

from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
from skellymodels.experimental.model_redo.biomechanics.biomechanics_processor import BiomechanicsProcessor


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

BiomechanicsProcessor.process_human(human)
# pprint(human.aspects)

