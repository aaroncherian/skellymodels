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

# ---------- pick ONE source for all points (here: FreeMoCap) ----------
P = freemocap_joints.as_dict  # or joints.as_dict if you want purely Qualisys
left_hip  = sg3(P['left_hip'])
right_hip = sg3(P['right_hip'])
right_knee  = sg3(P['right_knee'])
right_ankle = sg3(P['right_ankle'])
right_toe   = sg3(P['right_foot_index'])

def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

# Long axes & foot AP
a = unit(right_ankle - right_knee)   # shank long axis
b = unit(right_toe   - right_ankle)  # foot A–P (ankle->toe)
# make +AP roughly forward
if np.nanmean(b[:, 1]) < 0: b = -b

# Choose a neutral window (standing / first ~100 frames)
neutral = slice(0, 100)

# Per-frame "floating hinge" then take a robust constant hinge from neutral
h_pf = unit(np.cross(a, b))                          # instantaneous hinge candidates
# Force right-leg lateral direction: +X

def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

neutral = slice(0, 100)  # adjust if needed

# --- constant hinge from neutral, then constrain to lateral–vertical plane ---
h0_free = unit(np.nanmedian(h_pf[neutral], axis=0))        # (3,)
if h0_free[0] < 0:  # force right-leg lateral (+X)
    h0_free = -h0_free

# zero AP component to avoid cross-talk; re-normalize
h0 = unit(np.array([h0_free[0], 0.0, h0_free[2]]))
# keep orientation consistent with the original
if np.dot(h0, h0_free) < 0:
    h0 = -h0

# --- fixed 2D basis in plane ⟂ hinge ---
def proj_plane_one(v, n):
    return v - np.dot(v, n) * n

a0  = np.nanmedian(a[neutral], axis=0)
u1  = unit(proj_plane_one(a0, h0))           # along projected shank in the plane
u2  = unit(np.cross(h0, u1))                 # 90° in the plane

def proj_plane(V, n):
    return V - np.sum(V * n, axis=1, keepdims=True) * n

a_p = proj_plane(unit(a), h0)
b_p = proj_plane(unit(b), h0)

phi_a = np.arctan2(np.sum(a_p * u2, axis=1), np.sum(a_p * u1, axis=1))
phi_b = np.arctan2(np.sum(b_p * u2, axis=1), np.sum(b_p * u1, axis=1))
theta_rad = np.unwrap(phi_b - phi_a)
ankle_flexion_deg = np.degrees(theta_rad)

# remove neutral bias
ankle_flexion_deg -= np.nanmean(ankle_flexion_deg[neutral])

# # optional sign to match Qualisys
# ref = np.asarray(qual['ankle_angle_r'])[:len(ankle_flexion_deg)]
# w = slice(0, min(200, len(ref), len(ankle_flexion_deg)))
# if np.corrcoef(ankle_flexion_deg[w], ref[w])[0, 1] < 0:
#     ankle_flexion_deg *= -1

# --- smooth to Qualisys-like bandwidth (fs=30 Hz -> Nyquist 15) ---
from scipy import signal
b_lp, a_lp = signal.butter(4, 6/15, 'low')
ankle_flexion_deg_filt = signal.filtfilt(b_lp, a_lp, ankle_flexion_deg)

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
freemocap_path = path_to_recording / "validation" / "mediapipe_dlc" / "mediapipe_dlc_ik_results.mot"
# fmc_blender = pd.read_csv(fmc_blender_path)
HEADER_ROWS  = 10
qual = load_mot(qual_path, HEADER_ROWS)
fmc = load_mot(freemocap_path, HEADER_ROWS)

import matplotlib.pyplot as plt
# ankle_flexion = ankle_angles[:, 1]  # flexion/extension is θ_z from Z–X–Y sequence

def center_on_window(x, win):
    x = np.asarray(x)
    mu = np.nanmean(x[win])
    return x - mu

def unit2(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)

# same source for all points (e.g., FreeMoCap) + your sg3 smoothing
rk = sg3(freemocap_joints.as_dict['right_knee'])[:, [1,2]]   # Y,Z
ra = sg3(freemocap_joints.as_dict['right_ankle'])[:, [1,2]]
rt = sg3(freemocap_joints.as_dict['right_foot_index'])[:, [1,2]]

# shank vector in YZ (knee->ankle) and a "midfoot" to reduce MTP
s = unit2(ra - rk)
alpha = 0.6                                  # 0.5–0.7 works well
mid = ra + alpha*(rt - ra)
f = unit2(mid - ra)

# 2D signed angle in the sagittal plane
# θ = atan2( det([s,f]), dot(s,f) ), where det = s_y*f_z - s_z*f_y
det = s[:,0]*f[:,1] - s[:,1]*f[:,0]
dot = np.clip(np.sum(s*f, axis=1), -1.0, 1.0)
ankle_df_deg = np.degrees(np.arctan2(det, dot))

# neutral offset removal + optional sign flip to match Qualisys
neutral = slice(0, 100)
ankle_df_deg -= np.nanmean(ankle_df_deg[neutral])

ref = np.asarray(qual['ankle_angle_r'])[:len(ankle_df_deg)]
w = slice(0, min(200, len(ref), len(ankle_df_deg)))
if np.corrcoef(ankle_df_deg[w], ref[w])[0,1] < 0:
    ankle_df_deg *= -1

# mild low-pass to match IK smoothness (fs=30 -> 6 Hz)
from scipy import signal
b_lp, a_lp = signal.butter(4, 6/15, 'low')
ankle_df_deg = signal.filtfilt(b_lp, a_lp, ankle_df_deg)

plt.figure(figsize=(10, 4))
plt.plot(center_on_window(ankle_df_deg, slice(0, 100)), label='Ankle Flexion/Extension (deg)', color='blue')
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
ax1.plot(center_on_window(ankle_df_deg, slice(0, 100)), label='Ankle Flexion/Extension (deg)', color='blue')
ax1.plot(center_on_window(qual['ankle_angle_r'], slice(0, 100)), label='Qualisys', color='black', alpha=.5, linestyle='-.')
ax1.set_ylabel('Angle (degrees)')
ax1.set_title('Ankle Flexion/Extension Over Time')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(-25, 50)

# Bottom plot - Toe and heel Y trajectories
ax2.plot(right_toe[:, 0], label='Right Toe (Y)', color='red', alpha=0.7)
# ax2.plot(right_heel[:, 0], label='Right Heel (Y)', color='green', alpha=0.7)
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