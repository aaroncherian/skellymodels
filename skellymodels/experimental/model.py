from dataclasses import dataclass
from typing import Any

from skellymodels.model_info.qualisys_model_info import QualisysModelInfo

@dataclass
class AspectInfo:
    name: str
    value: Any

class Aspect:
    def __init__(self, name: str, landmark_names: list):
        self.name = name
        self.landmark_names = landmark_names
        self.aspect_info = {}

    def add_info(self, aspect_info: AspectInfo):
        self.aspect_info[aspect_info.name] = aspect_info.value

    def __getitem__(self, key: str):
        return self.aspect_info[key]

class Character:
    def __init__(self, name: str):
        self.name = name
        self.aspects = {}

    def add_aspect(self, aspect: Aspect):
        self.aspects[aspect.name] = aspect

    def __getitem__(self, key: str):
        return self.aspects[key]
    
    def __str__(self):
        return self.name



landmark_names = [
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
    "right_heel",
    "left_heel",
    "right_foot_index",
    "left_foot_index"
    ]

skeleton = Character(name="mediapipe")

body = Aspect(name="body", landmark_names=landmark_names)



skeleton.add_aspect(body)

model_info = QualisysModelInfo()

for key in model_info.to_dict().keys():
    aspect_info = AspectInfo(key, model_info.to_dict()[key])
    body.add_info(aspect_info)


f = 2

