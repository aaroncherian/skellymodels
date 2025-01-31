from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.models.aspect import Aspect

from skellymodels.experimental.model_redo.biomechanics.calculate_center_of_mass import calculate_center_of_mass
from skellymodels.experimental.model_redo.biomechanics.enforce_rigid_bones import enforce_rigid_bones, 


class BiomechanicsProcessor:

    @staticmethod
    def calculate_center_of_mass_from_aspect(aspect:Aspect):
        if aspect.anatomical_structure.center_of_mass_definitions:
            print('Calculating center of mass for aspect:', aspect.name)

            trajectory = aspect.trajectories['3d_xyz']

            total_body_com, segment_com = calculate_center_of_mass(
                segment_positions=trajectory.segment_data,
                center_of_mass_definitions=aspect.anatomical_structure.center_of_mass_definitions,
                num_frames=trajectory.num_frames
            )

            aspect.add_total_body_center_of_mass(total_body_center_of_mass=total_body_com)
            aspect.add_segment_center_of_mass(segment_center_of_mass=segment_com)
        
        else:
            print('Skipping center of mass calculation for aspect:', aspect.name)

    @staticmethod
    def enforce_rigid_bones_from_aspect(aspect:Aspect):
        if aspect.anatomical_structure.joint_hierarchy:
            
            trajectory = aspect.trajectories['3d_xyz']

            rigid_marker_data = enforce_rigid_bones(
                marker_trajectories=trajectory.data,
                segment_3d_positions=trajectory.segment_data,
                segment_conections=trajectory._segment_connections, #segment connections could also come from anatomical structures. there's some figuring out to do here
                joint_hierarchy= aspect.anatomical_structure.joint_hierarchy

            )

            aspect.add_trajectory(
                name='rigid_3d_xyz', data=rigid_marker_data, marker_names=aspect.anatomical_structure.marker_names
            )
        else:
            print('Skipping rigid bones enforcement for aspect:', aspect.name)

    @staticmethod
    def process_human(human: Human):
        for aspect in human.aspects.values():
            BiomechanicsProcessor.calculate_center_of_mass_from_aspect(aspect)
            BiomechanicsProcessor.enforce_rigid_bones_from_aspect(aspect)