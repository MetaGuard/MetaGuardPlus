from .saberMovementBuffer import SaberMovementBuffer
from .geometry import Plane, Vector3


class GoodCutEvent:
    def __init__(self, buffer: SaberMovementBuffer, note_orientation, cut_point=None):
        self.note_orientation = note_orientation
        self.buffer = buffer
        self.note_plane_was_cut = False
        self.finished = False
        last_added = buffer.get_curr()
        self.cut_plane_normal = last_added.cutPlaneNormal
        self.cut_time = last_added.time
        self.before_cut_rating = buffer.calculate_swing_rating()
        self.after_cut_rating = 0
        # self.note_plane = Plane(Vector3(0, 1, 0).rotate(note_orientation.rotation), note_orientation.position)
        self.note_plane = Plane(
            self.cut_plane_normal.cross(Vector3(0, 0, 1).rotate(note_orientation.rotation)),
            note_orientation.position
        )
        self.has_note_plane_been_cut = False

        cut_point = last_added.hiltPos if cut_point is None else cut_point
        self.cut_plane = Plane(self.cut_plane_normal, cut_point)
        self.note_forward = Vector3(0, 0, 1).rotate(
            self.note_orientation.rotation)

        self.calculate_acc()

    def update(self):
        curr_data = self.buffer.get_curr()
        prev_data = self.buffer.get_prev()
        if curr_data.time - self.cut_time > 0.4:
            self.finished = True
            return
        if prev_data is None:
            return

        if not self.has_note_plane_been_cut:
            self.note_plane.center = self.note_orientation.position
            self.note_plane.normal = self.cut_plane_normal.cross(self.note_forward)

        if self.note_plane.side(curr_data.tipPos) != self.note_plane.side(prev_data.tipPos):
            self.on_intersect_note_plane()
            self.has_note_plane_been_cut = True
        else:
            self.update_after_cut(curr_data)

    def on_intersect_note_plane(self):
        self.right_before = self.buffer.get_prev()
        self.right_after = self.buffer.get_curr()

        cut_hilt_pos = (self.right_before.hiltPos + self.right_after.hiltPos) / 2
        cut_tip_pos = self.note_plane.ray_trace(
            self.right_before.tipPos,
            self.right_after.tipPos - self.right_before.tipPos)[1]
        self.cut_time = self.right_after.time

        before_cut_error = (cut_tip_pos - cut_hilt_pos).angle(self.right_before.tipPos - self.right_before.hiltPos)
        after_cut_error = (cut_tip_pos - cut_hilt_pos).angle(self.right_after.tipPos - self.right_after.hiltPos)

        self.before_cut_rating = self.buffer.calculate_swing_rating(before_cut_error)
        self.after_cut_rating = after_cut_error / 60

    def update_after_cut(self, new_data):
        angle_with_normal = self.cut_plane_normal.angle(new_data.cutPlaneNormal)
        if angle_with_normal >= 90:
            self.finished = True
            return

        if angle_with_normal < 75:
            self.after_cut_rating += new_data.segmentAngle / 60
        else:
            self.after_cut_rating += new_data.segmentAngle * (90 - angle_with_normal) / 15 / 60

        if self.after_cut_rating > 1:
            self.after_cut_rating = 1
            self.finished = True

    # Might be incorrect
    def calculate_acc(self):
        max_cut_score = 15
        dist = self.cut_plane.dist_to_point(self.note_orientation.position)
        acc_percentage = 0 if dist > 0.3 else 1 - dist / 0.3
        self.acc = round(acc_percentage * max_cut_score)

    def get_score(self):
        return round(self.before_cut_rating * 70) + round(self.after_cut_rating * 30) + self.acc

    def get_score_breakdown(self):
        return (round(self.before_cut_rating * 70), round(self.after_cut_rating * 30), self.acc)
