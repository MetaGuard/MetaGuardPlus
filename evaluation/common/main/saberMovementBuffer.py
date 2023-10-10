from typing import *
from .typeDefs import SaberMovementData
from bsor.Bsor import VRObject
from .geometry import Vector3, Quaternion
from math import acos, pi

BUFFER_SIZE = 500


class SaberMovementBuffer:
    data: List[SaberMovementData]
    nextAddIndex: int

    def __init__(self):
        self.data = [None] * BUFFER_SIZE
        self.nextAddIndex = 0

    def get_curr(self) -> SaberMovementData:
        return self.data[(self.nextAddIndex - 1) % BUFFER_SIZE]

    def get_prev(self) -> SaberMovementData:
        return self.data[(self.nextAddIndex - 2) % BUFFER_SIZE]

    def add_saber_data(self, hand_object: VRObject, time: float):
        new_hilt_pos = Vector3(hand_object.x, hand_object.y, hand_object.z)
        new_tip_pos = new_hilt_pos + Vector3(0, 0, 1).rotate(
            Quaternion(hand_object.x_rot, hand_object.y_rot, hand_object.z_rot, hand_object.w_rot)
        )

        curr_data = self.get_curr()
        if curr_data is None:
            new_data = SaberMovementData(new_hilt_pos, new_tip_pos, None, None, time)
        else:
            new_data = SaberMovementData(new_hilt_pos, new_tip_pos, curr_data.hiltPos, curr_data.tipPos, time)

        self.data[self.nextAddIndex] = new_data
        self.nextAddIndex = (self.nextAddIndex + 1) % BUFFER_SIZE

    class BufferIterator:
        def __init__(self, buffer):
            self.buffer = buffer
            self.relativeIndex = 0

        def __next__(self) -> SaberMovementData:
            if self.relativeIndex >= BUFFER_SIZE:
                raise StopIteration

            output = self.buffer.data[(self.buffer.nextAddIndex - self.relativeIndex - 1) % BUFFER_SIZE]

            if output is None:
                raise StopIteration

            self.relativeIndex += 1
            return output

        def __iter__(self):
            return self

    def __iter__(self):
        return self.BufferIterator(self)

    def calculate_swing_rating(self, override=None):
        i = iter(self)
        first_data = next(i)
        first_normal = first_data.cutPlaneNormal
        first_time = first_data.time
        prev_time = first_time
        swing_rating = (first_data.segmentAngle if override is None else override) / 100

        for saber_data in i:
            if first_time - prev_time >= 0.4:
                break

            angle_with_normal = first_normal.angle(saber_data.cutPlaneNormal)
            if angle_with_normal >= 90:
                break

            prev_time = saber_data.time

            if angle_with_normal < 75:
                swing_rating += saber_data.segmentAngle / 100
            else:
                swing_rating += saber_data.segmentAngle * (90 - angle_with_normal) / 15 / 100

            if swing_rating > 1:
                return 1

        return swing_rating
