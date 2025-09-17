import numpy as np
from skellymodels.managers.human import Human
from skellymodels.models.trajectory import Trajectory
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def norm(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def get_shank_coordinate_system(joints:Trajectory):
    num_frames = joints.as_array.shape[0]
    left_hip = joints.as_dict['left_hip']
    right_hip = joints.as_dict['right_hip']
    right_knee = joints.as_dict['right_knee']
    right_ankle = joints.as_dict['right_ankle']

    hip_ml = norm(left_hip - right_hip)
    shank_z = norm(right_ankle - right_knee)
    shank_x = hip_ml
    shank_y = norm(np.cross(shank_z, shank_x))
    shank_x = norm(np.cross(shank_y, shank_z))

    R_shank = np.zeros((num_frames, 3, 3))
    for i in range(num_frames):
        R_shank[i] = np.column_stack([shank_x[i], shank_y[i], shank_z[i]])

    return R_shank

def get_foot_coordinate_system(joints:Trajectory):
    num_frames = joints.as_array.shape[0]
    ankle = joints.as_dict['right_ankle']
    toe = joints.as_dict['right_foot_index']
    heel = joints.as_dict['right_heel']
    
    y = toe - heel
    y_hat = norm(y)

    b = toe - ankle
    by = np.sum(b * y_hat, axis=1, keepdims=True)
    proj = by * y_hat

    x_raw = b - proj
    small = np.linalg.norm(x_raw, axis=1, keepdims=True) < 1e-9
    if np.any(small):
        g = np.tile(np.array([[1.0, 0.0, 0.0]]), (num_frames,1))
        g_proj = g - np.sum(g * y_hat, axis=1, keepdims=True) * y_hat
        x_raw[small[:,0]] = g_proj[small[:,0]]
    x_hat = norm(x_raw)

    z_hat = norm(np.cross(x_hat, y_hat))
    x_hat = norm(np.cross(y_hat, z_hat))

    R_foot = np.empty((num_frames, 3, 3))
    R_foot[:, :, 0] = x_hat
    R_foot[:, :, 1] = y_hat
    R_foot[:, :, 2] = z_hat

    return R_foot

def calculate_ankle_angles(human:Human, use_nonrigid = False):
    if use_nonrigid:
        joints = human.body.xyz
    else:
        joints = human.body.rigid_xyz
    num_frames = joints.as_array.shape[0]

    R_shank = get_shank_coordinate_system(joints)
    R_foot = get_foot_coordinate_system(joints)

    ankle_angles = np.zeros((num_frames, 3))

    R_rel = np.empty_like(R_shank)
    for i in range(num_frames):
        R_rel[i] = R_shank[i].T @ R_foot[i]

    r = R.from_matrix(R_rel)
    ankle_angles = r.as_euler('ZXY', degrees=True)

    return ankle_angles  # (num_frames, 3) in degrees, columns are [theta_z, theta_x, theta_y]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd


    def normalize_gait_cycle(gait_cycle:np.ndarray, n_points:int=101) -> np.ndarray:
        num_frames = gait_cycle.shape[0]
        x = np.linspace(0, 10, num = num_frames)
        x_new = np.linspace(0, 10, num = n_points)
        
        return np.apply_along_axis(lambda v: np.interp(x_new, x, v), axis=0, arr=gait_cycle)

    def subtract_neutral(angles:np.ndarray, neutral_frames:range) -> np.ndarray:
        neutral_mean = np.mean(angles[neutral_frames], axis=0)
        return angles - neutral_mean

    path_to_recording = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-52-16_GMT-4_jsm_treadmill_2")
    path_to_freemocap_parquet = path_to_recording/'validation'/'mediapipe'
    path_to_gait_events = path_to_recording/'validation'/'qualisys'/'gait_events.csv'

    neutral_frames = range(120, 200)

    frame_range = range(3700,4300)   

    human:Human = Human.from_data(path_to_freemocap_parquet)
    ankle_angles = subtract_neutral(calculate_ankle_angles(human), neutral_frames)

    qualisys_human = Human.from_data(path_to_recording/'validation'/'qualisys')
    qualisys_human.calculate()    
    qualisys_ankle_angles = subtract_neutral(calculate_ankle_angles(qualisys_human, use_nonrigid = True), neutral_frames)
    gait_events = pd.read_csv(path_to_gait_events)
    
    mask = (
        (gait_events["foot"]=="right") &
        (gait_events["event"] == "heel_strike") 
    )

    heelstrikes = [x for x in gait_events.loc[mask, "frame"].to_list() if x in frame_range]
    
    heelstrike_slices = []
    for i in range(len(heelstrikes)-1):
        heelstrike_slices.append(slice(heelstrikes[i], heelstrikes[i+1]))
    f = 2

    ankle_angle_cycles = np.stack([normalize_gait_cycle(ankle_angles[s]) for s in heelstrike_slices])
    
    qualisys_ankle_angle_cycles = np.stack([normalize_gait_cycle(qualisys_ankle_angles[s]) for s in heelstrike_slices])

    mean_ankle_angle_cycle = np.mean(ankle_angle_cycles, axis=0)

    mean_qualisys_ankle_angle_cycle = np.mean(qualisys_ankle_angle_cycles, axis=0)



    plt.figure(figsize = (10,4))
    plt.plot(mean_ankle_angle_cycle[:, 1], label='freemocap', color = 'blue')
    plt.plot(mean_qualisys_ankle_angle_cycle[:, 1], label='qualisys', color='black', alpha = .5, linestyle = '-.')
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Angle (degrees)')
    plt.title('Mean Ankle Flexion/Extension Over Gait Cycle')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().set_ylim(-25, 50)
    plt.gca().set_xlim(0, 100)
    plt.show()
    # f = 2 
    # def load_mot(path: Path, n_header: int) -> pd.DataFrame:
    #     df = pd.read_csv(
    #         path,
    #         delim_whitespace=True,
    #         skiprows=n_header,
    #         comment="#",
    #         header=0,
    #     )
    #     df.rename(columns={df.columns[0]: "time"}, inplace=True)
    #     return df.set_index("time")
    
    # HEADER_ROWS  = 10
    # qual_path = path_to_recording / "validation" / "qualisys" / "qualisys_ik_results.mot"

    # # qual = load_mot(qual_path, HEADER_ROWS)

    # plt.figure(figsize = (10,4))
    # plt.plot(ankle_angles[:, 1], label='freemocap', color = 'blue')
    # plt.plot(qualisys_ankle_angles[:, 1], label='qualisys', color='black', alpha = .5, linestyle = '-.')
    # # plt.plot(np.asarray(qual['ankle_angle_r']), label='Qualisys IK (.mot)', color='orange', alpha = .5, linestyle = '--')
    # plt.xlabel('Frame')
    # plt.ylabel('Angle (degrees)')
    # plt.title('Ankle Flexion/Extension Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.gca().set_ylim(-25, 50)

    # plt.show()
