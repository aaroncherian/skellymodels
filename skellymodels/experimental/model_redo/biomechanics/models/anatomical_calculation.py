from abc import ABC, abstractmethod
from skellymodels.experimental.model_redo.models.aspect import Aspect
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class CalculationResult:
    success: bool
    data: Dict[str, Any]
    messages: List[str]

class AnatomicalCalculation(ABC):
    @abstractmethod
    def calculate(self, aspect:Aspect) -> CalculationResult:
        """Perform the calculation and return results"""
        pass

    @abstractmethod
    def store(self, aspect:Aspect, results: CalculationResult):
        """Store the calculation results in the aspect"""
        pass

    def calculate_and_store(self, aspect:Aspect):
        "Perform calculation and store results in aspect"
        results = self.calculate(aspect)

        if results.success:
            self.store(
                aspect = aspect,
                results=results)

        return results
    