from abc import ABC, abstractmethod
from skellymodels.models.aspect import Aspect

from skellymodels.biomechanics.calculations.calculate_center_of_mass import calculate_center_of_mass
from skellymodels.biomechanics.calculations.enforce_rigid_bones import enforce_rigid_bones
from skellymodels.biomechanics.models.anatomical_calculation import AnatomicalCalculation, CalculationResult


class CenterOfMassCalculation(AnatomicalCalculation):
    def calculate(self, aspect:Aspect) -> CalculationResult:
        if not aspect.anatomical_structure.center_of_mass_definitions:
            return CalculationResult(
                success = False,
                data = {},
                messages=[f'No COM definitions for aspect: {aspect.name}, skipping COM calculation']
            )

        trajectory = aspect.trajectories['3d_xyz'] #NOTE: maybe put this in a try/except loop where the except also returns a CalcResult? with success=False

        total_body_com, segment_com = calculate_center_of_mass(
            segment_positions=trajectory.segment_data,
            center_of_mass_definitions=aspect.anatomical_structure.center_of_mass_definitions,
            num_frames=trajectory.num_frames
        )

        return CalculationResult(
            success = True,
            data = {
                'total_body_com': total_body_com,
                'segment_com': segment_com
            },
            messages=[f'Successfully calculated COM for aspect: {aspect.name}'] 
        )
    
    def store(self, aspect: Aspect, results: CalculationResult):
        if not results.success:
            return
        
        aspect.add_segment_center_of_mass(results.data['segment_com'])
        aspect.add_total_body_center_of_mass(results.data['total_body_com'])
        f = 2


    


class RigidBonesEnforcement(AnatomicalCalculation):
    def calculate(self, aspect:Aspect) -> CalculationResult:
        if not aspect.anatomical_structure.joint_hierarchy:
            return CalculationResult(
                success = False,
                data = {},
                messages = [f'No joint hierarchy defined for aspect: {aspect.name}, skipping rigid bones enforcement']
            )

        print('Enforcing rigid bones for aspect:', aspect.name)
        
        trajectory = aspect.trajectories['3d_xyz']

        rigid_marker_data = enforce_rigid_bones(
            marker_trajectories=trajectory.data,
            joint_hierarchy= aspect.anatomical_structure.joint_hierarchy
        )

        return CalculationResult(
            success=True,
            data = {'rigid_bones': rigid_marker_data},
            messages= [f'Successfully enforced rigid bones for aspect: {aspect.name}']
        )

    def store(self, aspect:Aspect, results: CalculationResult):
        if not results.success:
            return
        
        aspect.add_rigid_body_data(
            rigid_body_data=results.data['rigid_bones']
        )



class CalculationPipeline:
    def __init__(self, tasks: list[AnatomicalCalculation]):
        self.tasks = tasks

    def run(self, aspect:Aspect):
        """Instantiate tasks and run calculate_and_store on the given aspect."""
        results_log = []
        for task_cls in self.tasks:
            task_instance = task_cls()  
            results = task_instance.calculate_and_store(aspect)

            if results:
                results_log.extend(results.messages)

        return results_log



STANDARD_PIPELINE = CalculationPipeline([CenterOfMassCalculation, RigidBonesEnforcement])