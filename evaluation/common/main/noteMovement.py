# As of this commit, a lot needs to change in this code. A lot of
# function and variable definitions are incorrect or not filled in.
# Also, special care should be taken to cast between C# floats and doubles
# (in numpy 'single' and 'double') in the same way that the source does.

from .typeDefs import *
from typing import *
from math import cos, sin, pi as π, sqrt
from numpy import single
from .geometry import Vector3, Quaternion, Orientation
from bsor.Bsor import Bsor, Frame


def lerp_unclamped(a, b, t):
    return a + (b - a) * t


def lerp(a, b, t):
    if t < 0:
        t = 0
    if t > 1:
        t = 1
    return a + (b - a) * t


def quadratic_in_out(t):
    if t < 0.5:
        return 2 * t * t
    return (4 - 2 * t) * t - 1


def head_offset_z(noteInverseWorldRotation, headPseudoLocalPos):
    return (headPseudoLocalPos.rotate(noteInverseWorldRotation)).z


def get_z_pos(start, end, headOffsetZ, t):
    return lerp_unclamped(start + headOffsetZ * min(1, t * 2), end + headOffsetZ, t)


def move_towards_head(start, end, noteInverseWorldRotation, t, headPseudoLocalPos):
    headOffsetZ = head_offset_z(noteInverseWorldRotation, headPseudoLocalPos)
    return get_z_pos(start, end, headOffsetZ, t)


def quat_slerp(p, q, t):
    return Quaternion.Slerp(p, q, t)


def look_rotation(forwards, up):
    return Quaternion.from_forward_and_up(forwards, up)


class NoteData:
    def __init__(self, map: Map, note: Note):
        self.time = note.time * 60 / map.beatsPerMinute
        self.line_index = note.lineIndex
        self.flip_line_index = note.lineIndex
        self.flip_y_side = 0
        self.cut_direction_angle_offset = 0
        self.line_layer = note.lineLayer
        self.before_line_layer = 0
        self.note_type = note.type
        self.cut_direction = note.cutDirection
        self.gameplay_type = NoteData.GameplayType.NORMAL
        if note.type == 0:
            self.color_type = NoteData.ColorType.COLOR_A
        elif note.type == 0:
            self.color_type = NoteData.ColorType.COLOR_B
        else:
            self.color_type = NoteData.ColorType.NONE

    class GameplayType:
        NORMAL = 0

    class ColorType:
        NONE = -1
        COLOR_A = 0
        COLOR_B = 1

    class CutDirection:
        Up = 0
        Down = 1
        Left = 2
        Right = 3
        UpLeft = 4
        UpRight = 5
        DownLeft = 6
        DownRight = 7
        Any = 8
        NONE = 9


class MovementData:
    BEAT_OFFSET = 0
    JUMP_DURATION = 1
    move_speed = 200
    move_duration = 1
    center_pos = Vector3(0, 0, 0.65)

    def __init__(self, map: Map, note_data: NoteData, bsor: Bsor):
        self.note_lines_count = 4
        start_NJS = map.beatMaps[bsor.info.mode][bsor.info.difficulty].noteJumpMovementSpeed
        start_bpm = map.beatsPerMinute
        self.jump_duration = bsor.info.jumpDistance / start_NJS  # Notably NOT the same way the game calculates it
        self.right_vec = Vector3(1, 0, 0)
        forward_vec = Vector3(0, 0, 1)
        self.move_distance = self.move_duration * self.move_duration
        self.jump_distance = start_NJS * self.jump_duration
        self.move_end_pos = self.center_pos + forward_vec * (self.jump_distance * 0.5)
        self.jump_end_pos = self.center_pos - forward_vec * (self.jump_distance * 0.5)
        self.move_start_pos = self.center_pos + forward_vec * (self.move_distance + self.jump_distance * 0.5)
        self.spawn_ahead_time = self.move_duration + self.jump_duration * 0.5
        self.NJS = start_NJS
        self.JD = bsor.info.jumpDistance
        self.jumpOffsetY = self.get_y_offset_from_height(bsor.info.height)  # TODO: dynamic height
        self.end_rotation = self.get_rotation_angle(note_data.cut_direction) + note_data.cut_direction_angle_offset

        note_offset_1 = self.get_note_offset(note_data.line_index, note_data.before_line_layer)
        self.jump_end_pos += note_offset_1
        if note_data.color_type != NoteData.ColorType.NONE:
            note_offset_2 = self.get_note_offset(note_data.flip_line_index, note_data.before_line_layer)
            self.move_start_pos += note_offset_2
            self.move_end_pos += note_offset_2
        else:
            self.move_start_pos += note_offset_1
            self.move_end_pos += note_offset_1

        self.jump_gravity = self.get_gravity(note_data.line_layer, note_data.before_line_layer)

        self.z_offset = 0.25
        self.move_start_pos.z += self.z_offset
        self.move_end_pos.z += self.z_offset
        self.jump_end_pos.z += self.z_offset

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def get_y_offset_from_height(self, playerHeight):
        return self.clamp(((playerHeight - 1.7999999523162842) * 0.5), -0.2, 0.6)

    def get_note_offset(self, line_index, before_note_line_layer):
        return self.right_vec * ((-(self.note_lines_count - 1) * 0.5 + line_index) * 0.6) + Vector3(
            0, self.get_y_pos_from_layer(before_note_line_layer), 0)

    def get_y_pos_from_layer(self, layer):
        if (layer == 0):
            return 0.25
        if (layer == 1):
            return 0.85
        return 1.45

    def highest_jump_pos_y_for_line_layer(self, layer):
        if (layer == 0):
            return 0.85 + self.jumpOffsetY
        if (layer == 1):
            return 1.4 + self.jumpOffsetY
        return 1.9 + self.jumpOffsetY

    def get_gravity(self, lineLayer, beforeJumpLineLayer):
        num = (self.JD / self.NJS * 0.5)
        highest_pos = self.highest_jump_pos_y_for_line_layer(lineLayer)
        layer_height = self.get_y_pos_from_layer(beforeJumpLineLayer)
        return (2.0 * (highest_pos - layer_height) / (num * num))

    def get_rotation_angle(self, cut_direction):
        if cut_direction == NoteData.CutDirection.Up:
            return -180
        if cut_direction == NoteData.CutDirection.Down:
            return 0
        if cut_direction == NoteData.CutDirection.Left:
            return -90
        if cut_direction == NoteData.CutDirection.Right:
            return 90
        if cut_direction == NoteData.CutDirection.UpLeft:
            return -135
        if cut_direction == NoteData.CutDirection.UpRight:
            return 135
        if cut_direction == NoteData.CutDirection.DownLeft:
            return -45
        if cut_direction == NoteData.CutDirection.DownRight:
            return 45
        return 0


RANDOM_ROTATIONS = [
    Vector3(-0.9543871, -0.1183784, 0.2741019),
    Vector3(0.7680854, -0.08805521, 0.6342642),
    Vector3(-0.6780157, 0.306681, -0.6680131),
    Vector3(0.1255014, 0.9398643, 0.3176546),
    Vector3(0.365105, -0.3664974, -0.8557909),
    Vector3(-0.8790653, -0.06244748, -0.4725934),
    Vector3(0.01886305, -0.8065798, 0.5908241),
    Vector3(-0.1455435, 0.8901445, 0.4318099),
    Vector3(0.07651193, 0.9474725, -0.3105508),
    Vector3(0.1306983, -0.2508438, -0.9591639)
]
NUM_RANDOM_ROTATIONS = len(RANDOM_ROTATIONS)


def create_note_orientation_updater(map: Map, note: Note, bsor: Bsor):
    note_data = NoteData(map, note)
    movement_data = MovementData(map, note_data, bsor)
    movement_start_time = note_data.time - movement_data.move_duration - movement_data.jump_duration / 2
    jump_start_time = note_data.time - movement_data.jump_duration / 2
    move_duration = movement_data.move_duration
    jump_duration = movement_data.jump_duration
    floor_movement_start_pos = movement_data.move_start_pos
    floor_movement_end_pos = movement_data.move_end_pos
    jump_end_pos = movement_data.jump_end_pos
    gravity = movement_data.jump_gravity
    start_vertical_velocity = gravity * movement_data.jump_duration / 2
    y_avoidance = note_data.flip_y_side * 0.15 if note_data.flip_y_side <= 0 else note_data.flip_y_side * 0.45

    end_rotation = Quaternion.from_Euler(0, 0, movement_data.end_rotation)  # ###
    euler_angles = end_rotation.to_Euler()
    if note_data.gameplay_type == NoteData.GameplayType.NORMAL:
        # abs and % are in opposite order to that of the code since % is different in python vs C#
        index = abs(round(note_data.time * 10 + jump_end_pos.x * 2 + jump_end_pos.y * 2)) % NUM_RANDOM_ROTATIONS
        euler_angles += RANDOM_ROTATIONS[index] * 20
    middle_rotation = Quaternion.from_Euler(euler_angles.x, euler_angles.y, euler_angles.z)

    start_rotation = Quaternion(0, 0, 0, 1)
    rotate_towards_player = note_data.gameplay_type == NoteData.GameplayType.NORMAL
    world_rotation = Quaternion(0, 0, 0, 1)  # TBD how iportant this is yet
    inverse_world_rotation = Quaternion(0, 0, 0, 1)  # TBD how iportant this is yet
    world_to_player_rotation = Quaternion(0, 0, 0, 1)  # TBD how iportant this is yet
    end_distance_offset = 500

    def update(frame: Frame, object: Orientation):
        time = frame.time
        relative_time = time - movement_start_time  # Called num1 in source

        # Called floor movement in code
        if relative_time < move_duration:
            return lerp(floor_movement_start_pos, floor_movement_end_pos, relative_time / move_duration)

        relative_time = time - jump_start_time  # Called num1 in source
        start_pos = floor_movement_end_pos
        end_pos = jump_end_pos
        percentage_of_jump = relative_time / jump_duration  # Called t in source

        local_pos = Vector3(0, 0, 0)  # Called localPosition in source

        if start_pos.x == end_pos.x:
            local_pos.x = start_pos.x
        elif percentage_of_jump >= 0.25:
            local_pos.x = end_pos.x
        else:
            local_pos.x = lerp_unclamped(start_pos.x, end_pos.x, quadratic_in_out(percentage_of_jump * 4))

        local_pos.y = start_pos.y + start_vertical_velocity * relative_time - gravity * relative_time ** 2 * 0.5
        headPseudoLocalPos = Vector3(frame.head.x, frame.head.y, frame.head.z)
        local_pos.z = move_towards_head(start_pos.z, end_pos.z, inverse_world_rotation,
                                        percentage_of_jump, headPseudoLocalPos)

        if y_avoidance != 0 and percentage_of_jump < 0.25:
            local_pos.y += (0.5 - cos(percentage_of_jump * 8 * π) * 0.5) * y_avoidance

        if percentage_of_jump < 0.5:
            if percentage_of_jump >= 0.125:
                a = Quaternion.Slerp(middle_rotation, end_rotation, sin((percentage_of_jump - 0.125) * π * 2))
            else:
                a = Quaternion.Slerp(start_rotation, middle_rotation, sin((percentage_of_jump * π * 4)))

            if rotate_towards_player:
                head_pseudo_location = Vector3(frame.head.x, frame.head.y, frame.head.z)  # ###
                head_pseudo_location.y = lerp(head_pseudo_location.y, local_pos.y, 0.8)
                normalized = (local_pos - head_pseudo_location.rotate(inverse_world_rotation)).normal()
                rotated_object_up = Vector3(0, 1, 0).rotate(object.rotation)
                vector3 = rotated_object_up.rotate(world_to_player_rotation)
                b = look_rotation(normalized, vector3.rotate(inverse_world_rotation))
                object.rotation = Quaternion.Lerp(a, b, percentage_of_jump * 2)

            else:
                object.rotation = a

        if percentage_of_jump >= 0.75:
            num2 = (percentage_of_jump - 0.75) / 0.25
            local_pos.z -= lerp_unclamped(0, end_distance_offset, num2 * num2 * num2)

        object.position = local_pos.rotate(world_rotation)

    return update
