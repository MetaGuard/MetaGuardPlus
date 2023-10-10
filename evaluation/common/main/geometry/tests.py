import unittest
from geometry import Quaternion, Vector3


class TestQuaternionEulerAngles(unittest.TestCase):
    PRECISION = 4

    def assertVectorAlmostEqual(self, v1, v2, precision):
        self.assertAlmostEqual(v1.x, v2.x, precision)
        self.assertAlmostEqual(v1.y, v2.y, precision)
        self.assertAlmostEqual(v1.z, v2.z, precision)

    def assertQuaternionAlmostEqual(self, q1, q2, precision):
        self.assertAlmostEqual(q1.x, q2.x, precision)
        self.assertAlmostEqual(q1.y, q2.y, precision)
        self.assertAlmostEqual(q1.z, q2.z, precision)
        self.assertAlmostEqual(q1.w, q2.w, precision)

    def test_from_Euler(self):
        euler_angles = (319.45770, 295.00520, 10.19963)
        expected = Quaternion(-0.335855700, -0.476015200, -0.115075900, 0.804591800)

        calculated = Quaternion.from_Euler(*euler_angles)

        self.assertQuaternionAlmostEqual(expected, calculated, self.PRECISION)

    def test_to_Euler(self):
        quat = Quaternion(-0.335855700, -0.476015200, -0.115075900, 0.804591800)
        expected = Vector3(319.45770, 295.00520, 10.19963)

        calculated = quat.to_Euler()

        self.assertVectorAlmostEqual(expected, calculated, self.PRECISION)

    def test_forward_and_up(self):
        forward = Vector3(0.28664306476712226, -0.17795072543565926, 2.761551596672372)
        up = Vector3(-0.005308421083006178, 0.9997270086669968, -0.02275365481287681)
        expected = Quaternion(0.03203, 0.05162, -0.00019, 0.99815)

        calculated = Quaternion.from_forward_and_up(forward, up)

        self.assertQuaternionAlmostEqual(expected, calculated, self.PRECISION)

    def test_lerp(self):
        q1 = Quaternion(0.25446, 0.15489, 0.07744, 0.95145)
        q2 = Quaternion(0.61486, 0.64281, 0.41922, 0.18166)
        expected = Quaternion(0.44814, 0.38760, 0.23562, 0.77034)
        parameter = 0.37

        calculated = Quaternion.Lerp(q1, q2, parameter)

        self.assertQuaternionAlmostEqual(expected, calculated, self.PRECISION)

    def test_slerp(self):
        q1 = Quaternion(0.25446, 0.15489, 0.07744, 0.95145)
        q2 = Quaternion(0.61486, 0.64281, 0.41922, 0.18166)
        expected = Quaternion(0.45429, 0.39545, 0.24104, 0.76101)
        parameter = 0.37

        calculated = Quaternion.Slerp(q1, q2, parameter)

        self.assertQuaternionAlmostEqual(expected, calculated, self.PRECISION)


if __name__ == '__main__':
    unittest.main()
