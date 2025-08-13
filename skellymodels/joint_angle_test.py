from pathlib import Path
import numpy as np
from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import MediapipeModelInfo

path_to_recording = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2")
path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

data = np.load(path_to_data)

human:Human = Human.from_tracked_points_numpy_array(name="human_one",
                                                tracked_points_numpy_array=data,
                                                model_info=MediapipeModelInfo())


human.calculate()  # does our COM/Rigid bones calculations

left_hip = human.body.rigid_xyz.as_dict['left_hip']
right_hip = human.body.rigid_xyz.as_dict['right_hip']
left_knee = human.body.rigid_xyz.as_dict['left_knee']
left_ankle = human.body.rigid_xyz.as_dict['left_ankle']

left_thigh_vector = left_knee - left_hip    # (n_frames, 3)
left_shank_vector = left_ankle - left_knee  # (n_frames, 3)
hip_ml_vector = right_hip - left_hip        # (n_frames, 3)

thigh_norm = left_thigh_vector/ np.linalg.norm(left_thigh_vector, axis=1, keepdims=True)
shank_norm = left_shank_vector / np.linalg.norm(left_shank_vector, axis=1, keepdims=True)
hip_ml_norm = hip_ml_vector / np.linalg.norm(hip_ml_vector, axis=1, keepdims=True)


thigh_z = thigh_norm
thigh_x = hip_ml_norm

thigh_y = np.cross(thigh_z, thigh_x)
thigh_y = thigh_y / np.linalg.norm(thigh_y, axis=1, keepdims=True)

thigh_x = np.cross(thigh_y, thigh_z)
thigh_x = thigh_x / np.linalg.norm(thigh_x, axis=1, keepdims=True)

shank_z = shank_norm
shank_x = hip_ml_norm

shank_y = np.cross(shank_z, shank_x)
shank_y = shank_y / np.linalg.norm(shank_y, axis=1, keepdims=True)

shank_x = np.cross(shank_y, shank_z)
shank_x = shank_x / np.linalg.norm(shank_x, axis=1, keepdims=True)

R_thigh = np.zeros((data.shape[0], 3, 3))
R_shank = np.zeros((data.shape[0], 3, 3))

for i in range(data.shape[0]):
    R_thigh[i] = np.column_stack([thigh_x[i], thigh_y[i], thigh_z[i]])
    R_shank[i] = np.column_stack([shank_x[i], shank_y[i], shank_z[i]])


angles = np.zeros((data.shape[0], 3))

for i in range(data.shape[0]):

    R_relative = R_thigh[i].T @ R_shank[i]
    theta_x = np.arcsin(-R_relative[2, 1])
    theta_y = np.arctan2(R_relative[0, 1], R_relative[1, 1])
    theta_z = np.arctan2(R_relative[2, 0], R_relative[2, 2])

    angles[i] = np.rad2deg([theta_z, theta_x, theta_y])


left_ankle = human.body.rigid_xyz.as_dict['left_ankle']
left_toe = human.body.rigid_xyz.as_dict['left_foot_index']
left_heel = human.body.rigid_xyz.as_dict['left_heel']

foot_y = left_toe - left_heel
foot_y_norm = foot_y/np.linalg.norm(foot_y, axis=1, keepdims=True)
z_from_hip = np.cross(hip_ml_norm, foot_y)
foot_z_norm = z_from_hip / np.linalg.norm(z_from_hip, axis=1, keepdims=True)

foot_x = np.cross(foot_y_norm, foot_z_norm)
foot_x_norm = foot_x / np.linalg.norm(foot_x, axis=1, keepdims=True)
foot_y = np.cross(foot_z_norm, foot_x_norm)
foot_y_norm = foot_y / np.linalg.norm(foot_y, axis=1, keepdims=True)

R_foot = np.zeros((data.shape[0], 3, 3))
ankle_angles = np.zeros((data.shape[0], 3))  # Z-X-Y sequence

for i in range(data.shape[0]):
    R_foot[i] = np.column_stack([foot_x_norm[i], foot_y_norm[i], foot_z_norm[i]])

for i in range(data.shape[0]):
    R_relative = R_shank[i].T @ R_foot[i]
    theta_x = np.arcsin(-R_relative[2, 1])
    theta_y = np.arctan2(R_relative[0, 1], R_relative[1, 1])
    theta_z = np.arctan2(R_relative[2, 0], R_relative[2, 2])

    ankle_angles[i] = np.rad2deg([theta_z, theta_x, theta_y])


import pandas as pd
def load_mot(path: Path, n_header: int) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        skiprows=n_header,
        comment="#",
        header=0,
    )
    df.rename(columns={df.columns[0]: "time"}, inplace=True)
    return df.set_index("time")



fmc_blender_path = path_to_recording / "sweep_angles_all.csv"
qual_path = path_to_recording / "validation" / "qualisys" / "qualisys_ik_results.mot"
fmc_blender = pd.read_csv(fmc_blender_path)
HEADER_ROWS  = 10
qual = load_mot(qual_path, HEADER_ROWS)

import matplotlib.pyplot as plt
ankle_flexion = ankle_angles[:, 1]  # flexion/extension is θ_z from Z–X–Y sequence

def center_on_window(x, win):
    x = np.asarray(x)
    mu = np.nanmean(x[win])
    return x - mu

plt.figure(figsize=(10, 4))
plt.plot(center_on_window(ankle_flexion, slice(0, 100)), label='Ankle Flexion/Extension (deg)', color='blue')
plt.plot(center_on_window(fmc_blender['angle_angle#left_ankle_dorsiflexion_plantarflexion'], slice(0, 100)), label = 'fmc_blender', color='orange')
plt.plot(center_on_window(qual['ankle_angle_l'], slice(0, 100)), label='Qualisys', color='black', alpha = .5, linestyle = '-.')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Ankle Flexion/Extension Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# knee_flexion = angles[:, 1]  # flexion/extension is θ_z from Z–X–Y sequence

# plt.figure(figsize=(10, 4))
# plt.plot(knee_flexion, label='Knee Flexion/Extension (deg)', color='blue')
# plt.plot(fmc_blender['angle_angle#left_knee_extension_flexion'], label = 'fmc_blender', color='orange')
# plt.plot(range(len(qual)), qual['knee_angle_l']*-1, label='Qualisys', color='black', alpha = .5, linestyle = '-.')
# plt.xlabel('Frame')
# plt.ylabel('Angle (degrees)')
# plt.title('Knee Flexion/Extension Over Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

f = 2