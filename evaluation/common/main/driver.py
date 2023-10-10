from bsor.Bsor import Bsor
from .interpretMapFiles import Map
from .saberMovementBuffer import SaberMovementBuffer
from .noteManager import NoteManager
from .cutEvent import GoodCutEvent
from .scoreManager import ScoreManager
from .geometry import Vector3


def calculate_score_assuming_valid_times(map_data: Map, replay: Bsor):
    left_hand_buffer = SaberMovementBuffer()
    right_hand_buffer = SaberMovementBuffer()
    note_manager = NoteManager(map_data, replay)
    score_manager = ScoreManager()

    note_events = replay.notes[::-1]

    count = 0

    for frame in replay.frames[1:]:
        count += 1

        left_hand_buffer.add_saber_data(frame.left_hand, frame.time)
        right_hand_buffer.add_saber_data(frame.right_hand, frame.time)

        while len(note_events) > 0 and note_events[-1].event_time < frame.time:
            event = note_events.pop()
            note_object = note_manager.get_active_note_by_id(event.note_id)
            if note_object is None:
                continue

            note_object.handle_cut()
            cut_point = Vector3(*event.cut.cutPoint)
            if event.cut.saberType == 0:
                score_manager.register_cut_event(GoodCutEvent(left_hand_buffer, note_object.orientation, cut_point))
            else:
                score_manager.register_cut_event(GoodCutEvent(right_hand_buffer, note_object.orientation, cut_point))

        note_manager.update(frame)
        score_manager.update(frame)

    score_manager.finish()

    return score_manager.get_score(), score_manager.cut_events
