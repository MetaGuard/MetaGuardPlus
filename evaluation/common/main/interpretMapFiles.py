from .typeDefs import *
from json import load
from os import listdir
from .geometry import Vector3, Quaternion


def create_map(folderPath):
    if folderPath[-1] != "/":
        folderPath += "/"

    with open(folderPath + "info.dat", "r") as f:
        info_data = load(f)
        f.close()

    map = Map()
    map.beatsPerMinute = info_data['_beatsPerMinute']
    map.beatMaps = {}

    for beat_map_set in info_data['_difficultyBeatmapSets']:
        characteristic = beat_map_set['_beatmapCharacteristicName']
        difficulties = {}

        for diff in beat_map_set['_difficultyBeatmaps']:
            beat_map = BeatMap()
            beat_map.difficulty = diff['_difficulty']
            beat_map.noteJumpMovementSpeed = diff['_noteJumpMovementSpeed']
            beat_map.noteJumpStartBeatOffset = diff['_noteJumpStartBeatOffset']
            beat_map.notes = []
            beat_map.obstacles = []

            beat_map_file = open(folderPath + diff['_beatmapFilename'], "r")
            populate_beat_map(load(beat_map_file), beat_map)
            beat_map_file.close()

            difficulties[diff['_difficulty']] = beat_map

        map.beatMaps[characteristic] = difficulties
    return map


def populate_beat_map(JSO, beat_map: BeatMap):
    for note_data in JSO['_notes']:
        note = Note()
        note.time = note_data['_time']
        note.type = note_data['_type']
        note.lineIndex = note_data['_lineIndex']
        note.lineLayer = note_data['_lineLayer']
        note.cutDirection = note_data['_cutDirection']

        beat_map.notes.append(note)
    beat_map.notes.sort(key=lambda n: n.time)

    for obstacle_data in JSO['_obstacles']:
        obstacle = Obstacle()
        obstacle.time = obstacle_data['_time']
        obstacle.type = obstacle_data['_type']
        obstacle.lineIndex = obstacle_data['_lineIndex']
        obstacle.duration = obstacle_data['_duration']
        obstacle.width = obstacle_data['_width']

        beat_map.obstacles.append(obstacle)
    beat_map.obstacles.sort(key=lambda o: o.time)


class OrientationSeries:
    times: List[float]
    positions: List[Vector3]
    rotations: List[Quaternion]
    note_id: str
    length: int


def load_note_movement_data(folder_path: str):
    notes: List[OrientationSeries] = []
    for file_name in listdir(folder_path):
        orientation_series = OrientationSeries()
        orientation_series.times = []
        orientation_series.positions = []
        orientation_series.rotations = []
        orientation_series.note_id = file_name.split("_")[1]
        orientation_series.length = 0
        notes.append(orientation_series)

        with open(folder_path + file_name, "r") as f:
            lines = f.readlines()
            f.close()

        for line in lines[1:]:
            entries = (*map(float, line.split(",")),)
            orientation_series.times.append(entries[0])
            orientation_series.positions.append(Vector3(*entries[1:4]))
            orientation_series.rotations.append(Quaternion(*entries[4:]))
            orientation_series.length += 1

    notes.sort(key=lambda n: n.times[0])
    return notes
