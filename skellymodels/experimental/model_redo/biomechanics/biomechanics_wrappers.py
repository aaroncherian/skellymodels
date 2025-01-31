from skellymodels.experimental.model_redo.biomechanics.calculate_center_of_mass import (
    calculate_center_of_mass
)
from skellymodels.experimental.model_redo.biomechanics.enforce_rigid_bones import (
    calculate_bone_lengths_and_statistics,
    enforce_rigid_bones
)
# from skellymodels.experimental.model_redo.models.aspect import Aspect


# def calculate_center_of_mass_from_aspect(aspect: Aspect):
#     """Calculate total body and segment CoM and add to the aspect."""
#     if not aspect.anatomical_structure or not aspect.anatomical_structure.center_of_mass_definitions:
#         print(f'Missing center of mass definitions for aspect {aspect.name}, skipping CoM calculation')
#         return
    
#     print(f'Calculating center of mass for aspect: {aspect.name}')

#     trajectory = aspect.trajectories['3d_xyz']

#     total_body_com, segment_com = calculate_center_of_mass(
#         segment_positions=trajectory.segment_data,
#         center_of_mass_definitions=aspect.anatomical_structure.center_of_mass_definitions,
#         num_frames=trajectory.num_frames

#     )

#     aspect.add_total_body_center_of_mass(total_body_com)
#     aspect.add_segment_center_of_mass(segment_com)


# for aspect in human.aspects.values():
#     if aspect.anatomical_structure.center_of_mass_definitions:
#         print('Calculating center of mass for aspect:', aspect.name)
#         total_body_com, segment_com = calculate_center_of_mass_from_trajectory(aspect.trajectories['3d_xyz'], aspect.anatomical_structure.center_of_mass_definitions)

#         aspect.add_total_body_center_of_mass(total_body_center_of_mass=total_body_com)
#         aspect.add_segment_center_of_mass(segment_center_of_mass=segment_com)
    
#     else:
#         print('Skipping center of mass calculation for aspect:', aspect.name)
# pprint([human.aspects.values()])




# def enforce_rigid_bones(aspect: Aspect):
#     """Apply rigid bone enforcement to the trajectory data in the aspect."""
#     if not aspect.anatomical_structure or not aspect.anatomical_structure.joint_hierarchy:
#         print(f'Missing segment connections for aspect {aspect.name}, skipping rigid bone enforcement')
#         return

#     print(f'Enforcing rigid bones for aspect: {aspect.name}')

#     trajectory = aspect.trajectories['3d_xyz']
#     joint_hierarchy = aspect.anatomical_structure.joint_hierarchy
    
#     bone_lengths_and_statistics = calculate_bone_lengths_and_statistics(
#         marker_data=trajectory.data, 
#         segment_data=trajectory.segment_data
#     )

#     rigid_marker_data = enforce_rigid_bones(
#         marker_data=trajectory.data,
#         segment_connections=trajectory._segment_connections,
#         bone_lengths_and_statistics=bone_lengths_and_statistics,
#         joint_hierarchy=joint_hierarchy,
#     )

#     aspect.add_trajectory(
#         name='rigid_3d_xyz', data=rigid_marker_data, marker_names=aspect.anatomical_structure.marker_names
#     )

