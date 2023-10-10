from typing import *
from .geometry import Vector3


class Obstacle:
    width: int
    lineIndex: int
    time: float
    type: int       # Not sure what the different type of walls mean?
    duration: float


class Note:
    time: float
    type: int
    lineIndex: int
    lineLayer: int
    cutDirection: int


class BeatMap:
    difficulty: str
    noteJumpMovementSpeed: float
    noteJumpStartBeatOffset: float
    notes: List[Note]
    obstacles: List[Obstacle]


class Map:
    beatsPerMinute: int
    beatMaps: Dict[str, Dict[str, BeatMap]]


class SaberMovementData:
    hiltPos: Vector3
    tipPos: Vector3
    cutPlaneNormal: Vector3
    time: float
    segmentAngle: float

    def __init__(self, hilt, tip, prevHilt, prevTip, time):
        self.hiltPos = hilt
        self.tipPos = tip
        self.time = time
        if prevHilt is None or prevTip is None:
            self.cutPlaneNormal = None
            self.segmentAngle = 0
        else:
            self.cutPlaneNormal = (tip - hilt).cross((prevHilt + prevTip) / 2 - hilt).normal()
            self.segmentAngle = (tip - hilt).angle(prevTip - prevHilt)
