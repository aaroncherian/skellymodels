from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.biomechanics.biomechanics_wrappers import (AnatomicalCalculation,
                                                                                     CenterOfMassCalculation, 
                                                                                     RigidBonesEnforcement)


task_dictionary: dict[str, AnatomicalCalculation] = {
    'center_of_mass': CenterOfMassCalculation,
    'rigid_bones': RigidBonesEnforcement
}

class BiomechanicsProcessor:
    @staticmethod
    def process_human(human:Human):

        for aspect in human.aspects.values():
            results_log = []
            for task_name, task_function in task_dictionary.items():
                results = task_function.calculate(aspect=aspect)

                results_log.extend(results.messages)

                if results.success:
                    task_function.store(aspect=aspect,
                                        results = results)
            
            print(f"\nResults for aspect {aspect.name}:")
            for msg in results_log:
                print(f"  {msg}")

