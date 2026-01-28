from shapely.geometry import LineString
import math
import numpy as np

class RemoveNoise:
    def __init__(self, trajec_lst):
        self.trajec_lst = trajec_lst

    @staticmethod
    def distance_to_goal(start_x: float, start_y: float, end_x: float, end_y: float):
        goal_x = 105
        goal_y = 68/2

        diff_x_start = goal_x - start_x
        diff_y_start = abs(goal_y - start_y)

        start_distance_to_goal = np.sqrt(diff_x_start ** 2 + diff_y_start ** 2)
        diff_x_end = goal_x - end_x
        diff_y_end = abs(goal_y - end_y)
        end_distance_to_goal = np.sqrt(diff_x_end ** 2 + diff_y_end ** 2)

        return start_distance_to_goal - end_distance_to_goal


    def is_noise_trajectory(self, traj: list[list[float]]):
        """
        Determines if a given trajectory is considered noise based on a series of conditions.

        A trajectory (a list of [x, y] coordinates) is flagged as noise if any of the following conditions hold:
          1. The last coordinate is exactly [105, 0]
          2. The trajectory self-intersects (non-simple geometry)
          3. Endpoints or starting points have certain extreme values (e.g., y = 0 or 68, or x = 0)
          4. The trajectory starts in the opponent box but ends outside
          5. The trajectory both starts and ends in the own box
          6. The trajectory starts in own box, goes to the opponent half, then returns to own half
          7. Any coordinate equals [0, 0]
          8. Specific conditions based on several coordinates x-positions
          9. The sum of the segment distances is too short and the distance to goal is negative
          10. The Euclidean distance between the first and last coordinate is too small
        """
        # Condition 1: Last coordinate exactly [105, 0]
        if traj[-1][0] == 105 and traj[-1][1] == 0:
            return True

        # Condition 2: Check for self-intersection using Shapely
        coord_tuples = [tuple(coord) for coord in traj]
        line = LineString(coord_tuples)
        if not line.is_simple:  # Non-simple means it self-intersects
            return True

        # Condition 3: Undesired endpoint or starting y/x values
        if traj[-1][1] in (68, 0):
            return True
        if traj[-1][0] == 0:
            return True
        if traj[0][1] in (0, 68):
            return True

        # Condition 4: Trajectory starts in opponent box but ends outside.
        # Opponent box is defined as x between 88.5 and 105 and y between 13.85 and 54.5.
        if 88.5 <= traj[0][0] <= 105 and 13.85 <= traj[0][1] <= 54.5:
            if traj[-1][0] < 88.5 or traj[-1][1] < 13.85 or traj[-1][1] > 54.5:
                return True

        # Condition 5: Trajectory both starts and ends in own box.
        # Own box is defined as x between 0 and 16.5 and y between 13.85 and 54.5.
        if 0 <= traj[0][0] <= 16.5 and 13.85 <= traj[0][1] <= 54.5:
            if 0 <= traj[-1][0] <= 16.5 and 13.85 <= traj[-1][1] <= 54.5:
                return True

        # Condition 6: Trajectory starts in own box then goes to opponent half and back.
        if len(traj) > 2:
            if 0 <= traj[0][0] <= 11.5 and 13.85 <= traj[0][1] <= 54.5:
                if traj[1][0] > 52.5 and traj[2][0] < 34.5:
                    return True

        # Condition 7: Any coordinate equals [0, 0]
        for coord in traj:
            if coord == [0, 0]:
                return True

        # Conditions 8 through 17: Various checks based on several coordinate positions.
        # Note: Each of these conditions first checks if the trajectory has enough points.
        if len(traj) > 4 and traj[0][0] > 52.5:
            # Example: if the fourth coordinate (index 3) is in a specific range and the fifth (index 4) has a high y-value,
            # and the final x is less than the x-value at index 4.
            if 88.5 < traj[3][0] < 105 and 13.85 <= traj[3][1] <= 54.5:
                if traj[4][1] > 54.5 and traj[-1][0] < traj[4][0]:
                    return True

        if len(traj) > 3 and traj[0][0] <= 34.5:
            if traj[2][0] > 70.5 and traj[3][0] < 34.5:
                return True

        if len(traj) > 2 and traj[0][0] > 52.5:
            if traj[1][0] < 16.5 and traj[2][0] > 70.5:
                return True

        if len(traj) > 3 and traj[0][0] < 34.5:
            if traj[2][0] > 52.5 and traj[3][0] < traj[0][0]:
                return True

        if len(traj) > 4 and traj[0][0] > 70.5:
            if traj[3][0] < 16.5 and traj[4][0] > traj[2][0]:
                return True

        if len(traj) > 2 and traj[0][0] < 16.5:
            if traj[1][0] < traj[0][0] and traj[2][0] < traj[0][0]:
                return True

        if len(traj) > 2 and traj[0][0] > 88.5:
            if traj[2][0] < 16.5:
                return True

        if len(traj) > 4:
            if traj[1][0] < 16.5 and traj[4][0] > 88.5 and traj[-1][0] < 70.5:
                return True

        if traj[0][0] > 88.5 and traj[-1][0] < 16.5:
            return True

        if len(traj) > 4 and 52.5 < traj[0][0] < 70.5:
            if traj[3][0] > 88.5 and traj[4][0] < 34.5:
                return True

        if 16.5 < traj[0][0] < 34.5:
            if traj[-1][0] < 8:
                return True

        if len(traj) > 5 and 34.5 < traj[0][0] < 52.5:
            if traj[3][0] > 88.5 and traj[4][0] < 34.5 and traj[5][0] > 88.5:
                return True

        if len(traj) > 4 and traj[0][0] > 52.5:
            if traj[3][0] > 88.5 and traj[4][0] < 16.5:
                return True

        if len(traj) > 4 and traj[0][0] < 16.5:
            if traj[3][0] > 70.5 and traj[4][0] < 34.5:
                return True

        if len(traj) > 5 and traj[0][0] < 34.5:
            if traj[4][0] > 70.5 and traj[5][0] < traj[0][0]:
                return True

        if len(traj) > 3 and 70.5 < traj[0][0] < 88.5:
            if traj[2][0] < 16.5 and traj[3][0] > 88.5:
                return True

        if len(traj) > 4 and 52.5 < traj[0][0] < 70.5:
            if traj[3][0] > 88.5 and traj[4][0] < 52.5:
                return True

        if len(traj) == 8 and 52.5 < traj[0][0] < 70.5:
            if traj[-1][0] < 34.5:
                return True

        # Condition 18: Check if the total movement is very short and the "distance to goal" is negative.
        d = np.diff(traj, axis=0)
        seg_dists = np.sqrt((d ** 2).sum(axis=1))
        if seg_dists.sum() < 27 and self.distance_to_goal(traj[0][0], traj[0][1], traj[-1][0], traj[-1][1]) < 0:
            return True

        # Condition 19: If the Euclidean distance between the first and last points is very small.
        ddd = math.sqrt((traj[0][0] - traj[-1][0])**2 + (traj[0][1] - traj[-1][1])**2)
        if ddd < 3.5:
            return True

        return False

    def find_noise_indices(self):
        """
        Returns a list of indices corresponding to trajectories in coordinates_list that are considered noise.
        """
        noise_indices = []
        for idx, trajectory in enumerate(self.trajec_lst):
            if self.is_noise_trajectory(trajectory):
                noise_indices.append(idx)
        return noise_indices
    
    @staticmethod
    def remove_by_indices(iter, idxs: list[int]):
        idxs = set(idxs)
        return [e for i, e in enumerate(iter) if i not in idxs]
    
    def remove_noise(self):
        idx = self.find_noise_indices()
        self.trajec_lst = self.remove_by_indices(self.trajec_lst, idx)
        return self.trajec_lst