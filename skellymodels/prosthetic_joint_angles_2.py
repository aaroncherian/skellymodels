"""
Complete Ankle Angle Analysis Pipeline
Combines robust quaternion processing with proper Qualisys visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_closing, binary_opening
from skellymodels.managers.human import Human

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sg3(x, fps=30.0, win_sec=0.15, poly=3):
    """
    Savitzky-Golay smoothing for (n,3) arrays.
    """
    n = x.shape[0]
    win = int(max(poly + 2, round(win_sec * fps)))
    if win % 2 == 0: 
        win += 1
    win = min(win, n - 1 - (1 - (n % 2)))
    if win <= poly: 
        win = poly + 3
    if win % 2 == 0: 
        win += 1
    return savgol_filter(x, window_length=win, polyorder=poly, axis=0, mode="interp")


def load_mot(path: Path, n_header: int = 10) -> pd.DataFrame:
    """Load OpenSim .mot file"""
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        skiprows=n_header,
        comment="#",
        header=0,
    )
    df.rename(columns={df.columns[0]: "time"}, inplace=True)
    return df.set_index("time")


def center_on_window(x, win):
    """Center signal by subtracting mean of window"""
    x = np.asarray(x)
    mu = np.nanmean(x[win])
    return x - mu


def find_segments(binary_signal, value=True):
    """Find continuous segments where binary_signal == value."""
    segments = []
    start = None
    
    for i in range(len(binary_signal)):
        if binary_signal[i] == value and start is None:
            start = i
        elif binary_signal[i] != value and start is not None:
            segments.append((start, i))
            start = None
    
    if start is not None:
        segments.append((start, len(binary_signal)))
    
    return segments


# ============================================================================
# QUATERNION-BASED ANGLE CALCULATION
# ============================================================================

def robust_ankle_angles_quaternion(right_hip, right_knee, right_ankle, 
                                  right_toe, right_heel, left_hip):
    """
    Calculate ankle angles using quaternion-based method to avoid gimbal lock.
    """
    n_frames = right_ankle.shape[0]
    
    # Build anatomically consistent coordinate systems
    # Shank segment
    shank_long = right_ankle - right_knee
    shank_long = shank_long / np.linalg.norm(shank_long, axis=1, keepdims=True)
    
    # Medio-lateral reference
    ml_axis = left_hip - right_hip
    ml_axis = ml_axis / np.linalg.norm(ml_axis, axis=1, keepdims=True)
    
    # Shank coordinate system
    shank_y = np.cross(shank_long, ml_axis)
    shank_y = shank_y / np.linalg.norm(shank_y, axis=1, keepdims=True)
    shank_x = np.cross(shank_y, shank_long)
    shank_x = shank_x / np.linalg.norm(shank_x, axis=1, keepdims=True)
    
    # Foot coordinate system
    foot_long = right_toe - right_heel
    foot_long = foot_long / np.linalg.norm(foot_long, axis=1, keepdims=True)
    
    # Use shank's medio-lateral as initial reference
    foot_z = np.cross(shank_x, foot_long)
    foot_z = foot_z / np.linalg.norm(foot_z, axis=1, keepdims=True)
    foot_x = np.cross(foot_long, foot_z)
    foot_x = foot_x / np.linalg.norm(foot_x, axis=1, keepdims=True)
    
    # Calculate quaternions for each frame
    quaternions = np.zeros((n_frames, 4))
    ankle_angles = np.zeros((n_frames, 3))
    
    for i in range(n_frames):
        # Build rotation matrices
        R_shank = np.column_stack([shank_x[i], shank_y[i], shank_long[i]])
        R_foot = np.column_stack([foot_x[i], foot_long[i], foot_z[i]])
        
        # Relative rotation as quaternion
        R_relative = R_shank.T @ R_foot
        rotation = Rotation.from_matrix(R_relative)
        quaternions[i] = rotation.as_quat()  # [x, y, z, w]
        
        # Extract angles using stable quaternion-to-Euler conversion
        euler_angles = rotation.as_euler('ZXY', degrees=True)
        ankle_angles[i] = euler_angles
    
    # Apply quaternion smoothing
    smoothed_quaternions = quaternion_smooth(quaternions, window_size=5)
    
    # Convert back to angles
    smoothed_angles = np.zeros((n_frames, 3))
    for i in range(n_frames):
        rotation = Rotation.from_quat(smoothed_quaternions[i])
        smoothed_angles[i] = rotation.as_euler('ZXY', degrees=True)
    
    return smoothed_angles[:, 0], smoothed_angles[:, 1], smoothed_angles[:, 2], quaternions


def quaternion_smooth(quaternions, window_size=5):
    """
    Smooth quaternions using SLERP (Spherical Linear Interpolation).
    """
    n_frames = len(quaternions)
    smoothed = np.copy(quaternions)
    half_window = window_size // 2
    
    for i in range(half_window, n_frames - half_window):
        window_quats = quaternions[i-half_window:i+half_window+1]
        smoothed[i] = average_quaternions(window_quats)
    
    return smoothed


def average_quaternions(quaternions):
    """
    Average multiple quaternions using the method from Markley et al.
    """
    M = np.zeros((4, 4))
    for q in quaternions:
        q = q / np.linalg.norm(q)
        M += np.outer(q, q)
    
    M /= len(quaternions)
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg_quat = eigenvectors[:, -1]
    
    return avg_quat / np.linalg.norm(avg_quat)


# ============================================================================
# STANCE PHASE DETECTION AND ADAPTIVE FILTERING
# ============================================================================

def detect_stance_phase(ankle_velocity, heel_velocity, threshold_percentile=20):
    """
    Detect stance phase using kinematic data.
    """
    ankle_speed = np.linalg.norm(ankle_velocity, axis=1)
    heel_speed = np.linalg.norm(heel_velocity, axis=1)
    
    foot_speed = np.minimum(ankle_speed, heel_speed)
    threshold = np.percentile(foot_speed, threshold_percentile)
    stance_phase = foot_speed < threshold
    
    stance_phase = binary_closing(stance_phase, structure=np.ones(5))
    stance_phase = binary_opening(stance_phase, structure=np.ones(3))
    
    return stance_phase


def adaptive_filter(signal, stance_phase, fs=30, stance_cutoff=3, swing_cutoff=8):
    """
    Apply different filter cutoffs for stance and swing phases.
    """
    filtered = np.zeros_like(signal)
    
    b_stance, a_stance = butter(4, stance_cutoff / (fs/2), 'low')
    b_swing, a_swing = butter(4, swing_cutoff / (fs/2), 'low')
    
    stance_segments = find_segments(stance_phase, True)
    swing_segments = find_segments(~stance_phase, True)
    
    for start, end in stance_segments:
        if end - start > 10:
            pad_size = min(30, start, len(signal) - end)
            segment = signal[start-pad_size:end+pad_size]
            filtered_segment = filtfilt(b_stance, a_stance, segment)
            filtered[start:end] = filtered_segment[pad_size:-pad_size if pad_size > 0 else None]
        else:
            filtered[start:end] = signal[start:end]
    
    for start, end in swing_segments:
        if end - start > 10:
            pad_size = min(30, start, len(signal) - end)
            segment = signal[start-pad_size:end+pad_size]
            filtered_segment = filtfilt(b_swing, a_swing, segment)
            filtered[start:end] = filtered_segment[pad_size:-pad_size if pad_size > 0 else None]
        else:
            filtered[start:end] = signal[start:end]
    
    return filtered


def remove_spikes(signal, threshold=3, window_size=5):
    """
    Remove spikes using median absolute deviation (MAD) method.
    """
    median_filtered = medfilt(signal, kernel_size=window_size)
    deviation = np.abs(signal - median_filtered)
    mad = np.median(deviation)
    outliers = deviation > threshold * mad
    cleaned = np.copy(signal)
    cleaned[outliers] = median_filtered[outliers]
    return cleaned


def apply_anatomical_constraints(angles, max_dorsiflexion=25, max_plantarflexion=50):
    """
    Apply anatomical ROM constraints to ankle angles.
    """
    constrained = np.copy(angles)
    constrained[:, 0] = np.clip(constrained[:, 0], -max_plantarflexion, max_dorsiflexion)
    constrained[:, 1] = np.clip(constrained[:, 1], -20, 15)
    constrained[:, 2] = np.clip(constrained[:, 2], -15, 15)
    return constrained


# ============================================================================
# ORIGINAL METHOD FOR COMPARISON
# ============================================================================

def calculate_original_ankle_angles(right_hip, right_knee, right_ankle, 
                                   right_toe, right_heel, left_hip):
    """
    Your original ankle angle calculation method for comparison
    """
    n_frames = right_ankle.shape[0]
    
    # Build shank coordinate system
    right_shank_vector = right_ankle - right_knee
    hip_ml_vector = left_hip - right_hip
    
    shank_norm = right_shank_vector / np.linalg.norm(right_shank_vector, axis=1, keepdims=True)
    hip_ml_norm = hip_ml_vector / np.linalg.norm(hip_ml_vector, axis=1, keepdims=True)
    
    shank_z = shank_norm
    shank_x = hip_ml_norm
    
    shank_y = np.cross(shank_z, shank_x)
    shank_y = shank_y / np.linalg.norm(shank_y, axis=1, keepdims=True)
    
    shank_x = np.cross(shank_y, shank_z)
    shank_x = shank_x / np.linalg.norm(shank_x, axis=1, keepdims=True)
    
    # Build foot coordinate system (your original approach)
    foot_y = right_toe - right_heel
    foot_y_norm = foot_y / np.linalg.norm(foot_y, axis=1, keepdims=True)
    z_from_hip = np.cross(hip_ml_norm, foot_y)
    foot_z_norm = z_from_hip / np.linalg.norm(z_from_hip, axis=1, keepdims=True)
    
    foot_x = np.cross(foot_y_norm, foot_z_norm)
    foot_x_norm = foot_x / np.linalg.norm(foot_x, axis=1, keepdims=True)
    foot_y = np.cross(foot_z_norm, foot_x_norm)
    foot_y_norm = foot_y / np.linalg.norm(foot_y, axis=1, keepdims=True)
    
    R_shank = np.zeros((n_frames, 3, 3))
    R_foot = np.zeros((n_frames, 3, 3))
    ankle_angles = np.zeros((n_frames, 3))
    
    for i in range(n_frames):
        R_shank[i] = np.column_stack([shank_x[i], shank_y[i], shank_z[i]])
        R_foot[i] = np.column_stack([foot_x_norm[i], foot_y_norm[i], foot_z_norm[i]])
        
        R_relative = R_shank[i].T @ R_foot[i]
        theta_x = np.arcsin(-R_relative[2, 1])
        theta_y = np.arctan2(R_relative[0, 1], R_relative[1, 1])
        theta_z = np.arctan2(R_relative[2, 0], R_relative[2, 2])
        
        ankle_angles[i] = np.rad2deg([theta_z, theta_x, theta_y])
    
    # Apply same filtering as your original
    b, a = butter(4, 6/15, 'low')
    ankle_angles[:, 1] = filtfilt(b, a, ankle_angles[:, 1])
    
    return ankle_angles


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_ankle_angles_robust(freemocap_joints_dict):
    """
    Complete robust processing pipeline for ankle angles.
    Takes pre-loaded joint dictionary to avoid re-loading data.
    """
    print("Extracting joint positions...")
    left_hip = sg3(freemocap_joints_dict['left_hip'])
    right_hip = sg3(freemocap_joints_dict['right_hip'])
    right_knee = sg3(freemocap_joints_dict['right_knee'])
    right_ankle = sg3(freemocap_joints_dict['right_ankle'])
    right_toe = sg3(freemocap_joints_dict['right_foot_index'])
    right_heel = sg3(freemocap_joints_dict['right_heel'])
    
    print("Detecting stance phases...")
    ankle_velocity = np.gradient(right_ankle, axis=0)
    heel_velocity = np.gradient(right_heel, axis=0)
    stance_phase = detect_stance_phase(ankle_velocity, heel_velocity)
    
    print("Calculating angles using quaternion method...")
    ankle_df_pf, ankle_inv_ev, ankle_int_ext, quaternions = robust_ankle_angles_quaternion(
        right_hip, right_knee, right_ankle, right_toe, right_heel, left_hip
    )
    
    print("Removing spikes...")
    ankle_df_pf = remove_spikes(ankle_df_pf, threshold=3)
    ankle_inv_ev = remove_spikes(ankle_inv_ev, threshold=2)
    ankle_int_ext = remove_spikes(ankle_int_ext, threshold=2)
    
    print("Applying adaptive filtering...")
    ankle_df_pf_filtered = adaptive_filter(ankle_df_pf, stance_phase, 
                                          stance_cutoff=4, swing_cutoff=8)
    ankle_inv_ev_filtered = adaptive_filter(ankle_inv_ev, stance_phase,
                                           stance_cutoff=3, swing_cutoff=6)
    ankle_int_ext_filtered = adaptive_filter(ankle_int_ext, stance_phase,
                                            stance_cutoff=3, swing_cutoff=6)
    
    print("Applying anatomical constraints...")
    angles_final = apply_anatomical_constraints(
        np.column_stack([ankle_df_pf_filtered, ankle_inv_ev_filtered, ankle_int_ext_filtered])
    )
    
    print("Final smoothing pass...")
    b, a = butter(4, 6/15, 'low')
    angles_final[:, 0] = filtfilt(b, a, angles_final[:, 0])
    angles_final[:, 1] = filtfilt(b, a, angles_final[:, 1])
    angles_final[:, 2] = filtfilt(b, a, angles_final[:, 2])
    
    return {
        'dorsiflexion_plantarflexion': angles_final[:, 0],
        'inversion_eversion': angles_final[:, 1],
        'internal_external_rotation': angles_final[:, 2],
        'stance_phase': stance_phase,
        'quaternions': quaternions
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_ankle_comparison(results, qual_mot_path, original_angles=None, center_window=100):
    """
    Visualize ankle angles using the same approach as your original code
    """
    
    # Load Qualisys data
    HEADER_ROWS = 10
    qual = load_mot(qual_mot_path, HEADER_ROWS)
    
    # Get the processed angles
    processed_angles = results['dorsiflexion_plantarflexion']
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    
    # ========== SUBPLOT 1: Raw Comparison ==========
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(processed_angles, label='Robust (Quaternion + Adaptive)', color='blue', linewidth=2)
    
    if original_angles is not None:
        ax1.plot(original_angles, label='Original Method', color='red', linewidth=1.5, alpha=0.7)
    
    # Plot Qualisys - exactly like your original code
    ax1.plot(range(len(qual)), qual['ankle_angle_r'], label='Qualisys', 
             color='black', alpha=0.5, linestyle='-.', linewidth=1.5)
    
    # Highlight stance phases
    if 'stance_phase' in results:
        stance = results['stance_phase']
        stance_label_added = False
        for start, end in find_segments(stance, True):
            label = 'Stance Phase' if not stance_label_added else ''
            ax1.axvspan(start, end, alpha=0.15, color='gray', label=label)
            stance_label_added = True
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Ankle Dorsiflexion/Plantarflexion - Raw Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(len(processed_angles), len(qual)))
    
    # ========== SUBPLOT 2: Centered Comparison ==========
    ax2 = plt.subplot(3, 1, 2)
    
    win = slice(0, center_window)
    
    ax2.plot(center_on_window(processed_angles, win), 
             label='Robust (Quaternion + Adaptive)', color='blue', linewidth=2)
    
    if original_angles is not None:
        ax2.plot(center_on_window(original_angles, win), 
                label='Original Method', color='red', linewidth=1.5, alpha=0.7)
    
    # Qualisys - exactly like your original code
    ax2.plot(center_on_window(qual['ankle_angle_r'], win), 
             label='Qualisys', color='black', alpha=0.5, linestyle='-.', linewidth=1.5)
    
    if 'stance_phase' in results:
        for start, end in find_segments(stance, True):
            if start < center_window * 2:
                ax2.axvspan(start, end, alpha=0.15, color='gray')
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title(f'Ankle Flexion/Extension Over Time (Centered on first {center_window} frames)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-25, 50)
    ax2.set_xlim(0, min(len(processed_angles), len(qual)))
    
    # ========== SUBPLOT 3: Zoomed view ==========
    ax3 = plt.subplot(3, 1, 3)
    
    zoom_start = 200
    zoom_end = min(400, len(processed_angles), len(qual))
    zoom_slice = slice(zoom_start, zoom_end)
    frames = np.arange(zoom_start, zoom_end)
    
    zoom_win = slice(0, zoom_end - zoom_start)
    
    ax3.plot(frames, center_on_window(processed_angles[zoom_slice], zoom_win),
             label='Robust', color='blue', linewidth=2.5)
    
    if original_angles is not None and len(original_angles) > zoom_end:
        ax3.plot(frames, center_on_window(original_angles[zoom_slice], zoom_win),
                label='Original', color='red', linewidth=2, alpha=0.7)
    
    if len(qual) > zoom_end:
        qual_zoom = qual['ankle_angle_r'].iloc[zoom_slice]
        ax3.plot(frames, center_on_window(qual_zoom.values, zoom_win),
                label='Qualisys', color='black', linewidth=1.5, linestyle='-.', alpha=0.7)
    
    if 'stance_phase' in results:
        stance_zoom = stance[zoom_slice]
        for start, end in find_segments(stance_zoom, True):
            ax3.axvspan(zoom_start + start, zoom_start + end, alpha=0.15, color='gray')
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title(f'Zoomed View: Frames {zoom_start}-{zoom_end} (Centered)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-20, 20)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    
    min_len = min(len(processed_angles), len(qual))
    qual_values = qual['ankle_angle_r'].values[:min_len]
    proc_values = processed_angles[:min_len]
    
    rmse = np.sqrt(np.mean((proc_values - qual_values)**2))
    print(f"\nRobust Method vs Qualisys:")
    print(f"  RMSE: {rmse:.2f}°")
    print(f"  Mean difference: {np.mean(proc_values - qual_values):.2f}°")
    
    if original_angles is not None:
        orig_values = original_angles[:min_len]
        rmse_orig = np.sqrt(np.mean((orig_values - qual_values)**2))
        print(f"\nOriginal Method vs Qualisys:")
        print(f"  RMSE: {rmse_orig:.2f}°")
        print(f"  Improvement: {rmse_orig - rmse:.2f}° ({'better' if rmse < rmse_orig else 'worse'})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Set paths
    path_to_recording = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
    path_to_freemocap_parquet = path_to_recording / 'output_data' / 'mediapipe_dlc' / 'freemocap_data_by_frame.parquet'
    qual_mot_path = path_to_recording / "validation" / "qualisys" / "qualisys_ik_results.mot"
    
    print("="*60)
    print("ANKLE ANGLE ANALYSIS - ROBUST vs ORIGINAL")
    print("="*60)
    
    # Load FreeMoCap data
    print("\nLoading FreeMoCap data...")
    freemocap_human = Human.from_parquet(path_to_freemocap_parquet)
    freemocap_joints = freemocap_human.body.xyz
    
    # Calculate original angles for comparison
    print("\nCalculating original ankle angles...")
    original_angles = calculate_original_ankle_angles(
        sg3(freemocap_joints.as_dict['left_hip']),
        sg3(freemocap_joints.as_dict['right_hip']),
        sg3(freemocap_joints.as_dict['right_knee']),
        sg3(freemocap_joints.as_dict['right_ankle']),
        sg3(freemocap_joints.as_dict['right_foot_index']),
        sg3(freemocap_joints.as_dict['right_heel'])
    )
    
    # Process with robust method
    print("\nProcessing with robust quaternion method...")
    results = process_ankle_angles_robust(freemocap_joints.as_dict)
    
    # Visualize comparison
    print("\nCreating visualization...")
    visualize_ankle_comparison(
        results, 
        qual_mot_path, 
        original_angles=original_angles[:, 1],  # Using flexion/extension component
        center_window=100
    )
    
    # Save results
    output_path = path_to_freemocap_parquet.parent
    
    # Save both methods for comparison
    np.save(output_path / "ankle_angles_robust.npy", results)
    np.save(output_path / "ankle_angles_original.npy", original_angles)
    
    # Save as CSV
    df = pd.DataFrame({
        'robust_dorsiflexion_plantarflexion': results['dorsiflexion_plantarflexion'],
        'original_flexion_extension': original_angles[:, 1],
        'stance_phase': results['stance_phase'].astype(int)
    })
    df.to_csv(output_path / "ankle_angles_comparison.csv", index_label='frame')
    
    print(f"\nResults saved to {output_path}")
    print("\nAnalysis complete!")