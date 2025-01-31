from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.models.aspect import Aspect

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
            
            for task_name, task_function in task_dictionary.items():
                task_function.calculate(aspect=aspect)
