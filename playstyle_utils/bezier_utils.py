import numpy as np

class Bezier():
    def TwoPoints(t: float, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Returns a point between P1 and P2, parametised by t.
        """
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t: float, points: list) -> list:
        """
        Returns a list of points interpolated by the Bezier process
        """
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints

    def Point(t: float, points: list) -> np.ndarray:
        """
        Returns a point interpolated by the Bezier process
        """
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]

    def Curve(t_values: list, points: list) -> np.ndarray:
        """
        Returns a point interpolated by the Bezier process
        """
        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve