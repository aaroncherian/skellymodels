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
    @staticmethod
    @abstractmethod
    def calculate(aspect:Aspect) -> CalculationResult:
        """Perform the calculation and return results"""
        pass

    @staticmethod
    @abstractmethod
    def store(aspect:Aspect, results: CalculationResult):
        """Store the calculation results in the aspect"""
        pass

    