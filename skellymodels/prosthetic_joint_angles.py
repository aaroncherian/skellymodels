from pathlib import Path
import numpy as np
from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import MediapipeModelInfo

path_to_recording = Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")

path_to_data = path_to_recording/'validation'/'qualisys'/'freemocap_data_by_frame.parquet'

path_to_freemocap_parquet = path_to_recording/'validation'/'mediapipe'/'freemocap_data_by_frame.parquet'
# path_to_data = path_to_recording/'output_data'/'mediapipe_dlc'/'freemocap_data_by_frame.parquet'

# data = np.load(path_to_data)

# human:Human = Human.from_tracked_points_numpy_array(name="human_one",
#                                                 tracked_points_numpy_array=data,
#                                                 model_info=MediapipeModelInfo())

from scipy.signal import savgol_filter
import numpy as np

def sg3(x, fps=30.0, win_sec=0.15, poly=3):
    """
    Savitzky–Golay smoothing for (n,3) arrays.
    win_sec: ~0.10–0.20 s usually works well for foot markers.
    poly: 2–3 is typical; higher risks ringing.
    """
    n = x.shape[0]
    # choose odd window length, at least poly+2, and < n
    win = int(max(poly + 2, round(win_sec * fps)))
    if win % 2 == 0: win += 1
    win = min(win, n - 1 - (1 - (n % 2)))  # ensure odd < n
    if win <= poly: win = poly + 3  # final safety
    if win % 2 == 0: win += 1

    return savgol_filter(x, window_length=win, polyorder=poly, axis=0, mode="interp")

human:Human = Human.from_parquet(path_to_data)
human.calculate()
data = human.body.xyz.as_array
# human.calculate()  # does our COM/Rigid bones calculations

joints = human.body.xyz

freemocap_human:Human = Human.from_parquet(path_to_freemocap_parquet)
freemocap_joints = freemocap_human.body.xyz

left_hip = sg3(freemocap_joints.as_dict['left_hip'])
right_hip = sg3(freemocap_joints.as_dict['right_hip'])
right_knee = sg3(freemocap_joints.as_dict['right_knee'])
right_ankle = sg3(freemocap_joints.as_dict['right_ankle'])

right_thigh_vector = right_knee - right_hip    # (n_frames, 3)
right_shank_vector = right_ankle - right_knee  # (n_frames, 3)
hip_ml_vector = left_hip - right_hip        # (n_frames, 3)

thigh_norm = right_thigh_vector/ np.linalg.norm(right_thigh_vector, axis=1, keepdims=True)
shank_norm = right_shank_vector / np.linalg.norm(right_shank_vector, axis=1, keepdims=True)
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


right_ankle = sg3(freemocap_joints.as_dict['right_ankle'])
right_toe = sg3(freemocap_joints.as_dict['right_foot_index'])
right_heel = sg3(freemocap_joints.as_dict['right_heel'])

foot_y = right_toe - right_heel
foot_y_norm = foot_y/np.linalg.norm(foot_y, axis=1, keepdims=True)
z_from_hip = np.cross(hip_ml_norm, foot_y)
foot_z_norm = z_from_hip / np.linalg.norm(z_from_hip, axis=1, keepdims=True)

foot_x = np.cross(foot_y_norm, foot_z_norm)
foot_x_norm = foot_x / np.linalg.norm(foot_x, axis=1, keepdims=True)
foot_y = np.cross(foot_z_norm, foot_x_norm)
foot_y_norm = foot_y / np.linalg.norm(foot_y, axis=1, keepdims=True)

R_foot = np.zeros((data.shape[0], 3, 3))
ankle_angles = np.zeros((data.shape[0], 3))  # Z-X-Y sequence

# shank_vector_sagittal = shank_vector.copy()

for i in range(data.shape[0]):
    R_foot[i] = np.column_stack([foot_x_norm[i], foot_y_norm[i], foot_z_norm[i]])

for i in range(data.shape[0]):
    R_relative = R_shank[i].T @ R_foot[i]
    theta_x = np.arcsin(-R_relative[2, 1])
    theta_y = np.arctan2(R_relative[0, 1], R_relative[1, 1])
    theta_z = np.arctan2(R_relative[2, 0], R_relative[2, 2])

    ankle_angles[i] = np.rad2deg([theta_z, theta_x, theta_y])

def _norm(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n==0, 1.0, n)

# 1) Build shank and foot vectors
v_shank = right_ankle - right_knee           # tibia pointing ankle-ward
v_foot  = right_toe  - right_heel            # A–P along the foot

# 2) Project to sagittal plane (zero the ML/X component)
v_shank_sag = v_shank.copy(); v_shank_sag[:,0] = 0.0
v_foot_sag  = v_foot.copy();  v_foot_sag[:,0]  = 0.0

# 3) Normalize (avoid div-by-zero)
v_shank_sag = _norm(v_shank_sag)
v_foot_sag  = _norm(v_foot_sag)

# 4) Signed angle between them, about +X (right-hand rule)
dot = np.clip(np.sum(v_shank_sag * v_foot_sag, axis=1), -1.0, 1.0)
cross = np.cross(v_shank_sag, v_foot_sag)    # (n,3); its X component gives the sign
angle_rad = np.arctan2(cross[:,0], dot)      # signed angle in radians
ankle_df_sag_deg = np.degrees(angle_rad)   

from scipy import signal
import pandas as pd

# Apply a simple low-pass Butterworth filter to smooth the ankle angle signal
b, a = signal.butter(4, 6/15, 'low')  # 4th order, cutoff freq=6 Hz
ankle_df_sag_deg_filtered = signal.filtfilt(b, a, ankle_angles[:, 1] )
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
freemocap_path = path_to_recording / "validation" / "mediapipe_dlc" / "mediapipe_dlc_ik_results.mot"
# fmc_blender = pd.read_csv(fmc_blender_path)
HEADER_ROWS  = 10
qual = load_mot(qual_path, HEADER_ROWS)
fmc = load_mot(freemocap_path, HEADER_ROWS)

import matplotlib.pyplot as plt
ankle_flexion = ankle_angles[:, 1]  # flexion/extension is θ_z from Z–X–Y sequence

def center_on_window(x, win):
    x = np.asarray(x)
    mu = np.nanmean(x[win])
    return x - mu

plt.figure(figsize=(10, 4))
plt.plot(center_on_window(ankle_angles[:,1], slice(0, 100)), label='Ankle Flexion/Extension (deg)', color='blue')
# plt.plot(center_on_window(fmc_blender['angle_angle#right_ankle_dorsiflexion_plantarflexion'], slice(0, 100)), label = 'fmc_blender', color='orange')
plt.plot(center_on_window(qual['ankle_angle_r'], slice(0, 100)), label='Qualisys', color='black', alpha = .5, linestyle = '-.')
# plt.plot(center_on_window(fmc['ankle_angle_r'], slice(0, 100)), label='Freemocap', color='orange', alpha = .5, linestyle = '--')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Ankle Flexion/Extension Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().set_ylim(-25, 50)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top plot - Ankle angles
ax1.plot(center_on_window(ankle_flexion, slice(0, 100)), label='Ankle Flexion/Extension (deg)', color='blue')
ax1.plot(center_on_window(qual['ankle_angle_r'], slice(0, 100)), label='Qualisys', color='black', alpha=.5, linestyle='-.')
ax1.set_ylabel('Angle (degrees)')
ax1.set_title('Ankle Flexion/Extension Over Time')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(-25, 50)

# Bottom plot - Toe and heel Y trajectories
ax2.plot(right_toe[:, 0], label='Right Toe (Y)', color='red', alpha=0.7)
ax2.plot(right_heel[:, 0], label='Right Heel (Y)', color='green', alpha=0.7)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Y Position (mm)')
ax2.set_title('Toe and Heel Anterior-Posterior Trajectories')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# np.save(path_to_data.parent / "ankle_flexion_angles.npy", ankle_angles)

# import matplotlib.pyplot as plt
# knee_flexion = angles[:, 1]  # flexion/extension is θ_z from Z–X–Y sequence

# plt.figure(figsize=(10, 4))
# plt.plot(knee_flexion, label='Knee Flexion/Extension (deg)', color='blue')
# # plt.plot(fmc_blender['angle_angle#right_knee_extension_flexion'], label = 'fmc_blender', color='orange')
# plt.plot(range(len(qual)), qual['knee_angle_l']*-1, label='Qualisys', color='black', alpha = .5, linestyle = '-.')
# plt.xlabel('Frame')
# plt.ylabel('Angle (degrees)')
# plt.title('Knee Flexion/Extension Over Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# f = 2