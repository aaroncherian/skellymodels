from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.biomechanics.anatomical_calculations import STANDARD_PIPELINE




class BiomechanicsProcessor:
    @staticmethod
    def process_human(human: Human):
        for aspect in human.aspects.values():
            results_log = []

            tasks = STANDARD_PIPELINE.get_tasks()
            for task in tasks:
                results = task.calculate_and_store(aspect=aspect)

                if results:
                    results_log.extend(results.messages)

            print(f"\nResults for aspect {aspect.name}:")
            for msg in results_log:
                print(f"  {msg}")

