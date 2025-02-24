from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.biomechanics.anatomical_calculations import (AnatomicalCalculation,
                                                                                     CenterOfMassCalculation, 
                                                                                     RigidBonesEnforcement)


task_dictionary: dict[str, AnatomicalCalculation] = {
    'center_of_mass': CenterOfMassCalculation,
    'rigid_bones': RigidBonesEnforcement
}

class BiomechanicsProcessor:
    @staticmethod
    def process_human(human: Human):
        for aspect in human.aspects.values():
            results_log = []
            for task_name, TaskClass in task_dictionary.items():
                task_instance = TaskClass()  # Instantiate the class
                
                # Use the new method
                results = task_instance.calculate_and_store(aspect=aspect)
                
                if results:
                    results_log.extend(results.messages)

            print(f"\nResults for aspect {aspect.name}:")
            for msg in results_log:
                print(f"  {msg}")

