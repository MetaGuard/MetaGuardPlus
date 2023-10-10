from .interpretMapFiles import Map
from bsor.Bsor import Bsor
from .geometry import Orientation, Vector3, Quaternion
from .noteMovement import create_note_orientation_updater

class NoteObject:
    def __init__(self, map, note, replay, manager):
        self.id = 30000 + note.lineIndex*1000 + note.lineLayer*100 + note.type*10 + note.cutDirection
        self.updater = create_note_orientation_updater(map, note, replay)
        self.orientation = Orientation(Vector3(0, 0, 0), Quaternion(0, 0, 0, 1))
        self.manager = manager

    def update(self, frame):
        self.updater(frame, self.orientation)

    def handle_cut(self):
        self.manager.active.remove(self)


class NoteManager:
    def __init__(self, map_data: Map, replay: Bsor):
        self.beatmap = map_data.beatMaps[replay.info.mode][replay.info.difficulty]
        self.notes = self.beatmap.notes[::-1]
        self.map = map_data
        self.replay = replay
        self.spawn_ahead_time = 1 + self.replay.info.jumpDistance / self.beatmap.noteJumpMovementSpeed * 0.5
        self.active = []

    def update(self, frame):
        while len(self.notes) > 0 and frame.time >= self.get_spawn_time(self.notes[-1]):
            new_note = NoteObject(self.map, self.notes.pop(), self.replay, self)
            self.active.append(new_note)

        for note_object in self.active:
            note_object.update(frame)

    def get_spawn_time(self, note):
        return (note.time * 60 / self.map.beatsPerMinute - self.spawn_ahead_time)

    def get_active_note_by_id(self, id) -> NoteObject:
        for note_object in self.active:
            if note_object.id == id:
                return note_object
        return None
