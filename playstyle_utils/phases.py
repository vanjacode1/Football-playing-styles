import pandas as pd
import numpy as np

class SplitPossessionPhases:

    def _group_me(self, values: list[int]):
        values.sort()
        sublist = []
        while values:
            v = values.pop(0)
            if not sublist or sublist[-1] in [v, v-1]:
                sublist.append(v)
            else:
                yield sublist
                sublist = [v]
        if sublist:
            yield sublist

    def partition(self, values: list[int], indices: list[int]):
        idx = 0
        for index in indices:
            sublist = []
            while idx < len(values) and values[idx] < index:
                sublist.append(values[idx])
                idx += 1
            if sublist:
                yield sublist

    def convert_nice_time_to_seconds(self, nice_time_str):
        """
        Converts a time string (e.g., "3m40s") into total seconds
        """
        minutes, seconds = nice_time_str.replace("s", "").split("m")
        return int(minutes) * 60 + int(seconds)


    def add_time_seconds_column(self, df: pd.DataFrame):
        """
        Adds a new column 'time_seconds' to the DataFrame based on the 'nice_time' column
        """
        df = df.copy()
        df['time_seconds'] = df['nice_time'].apply(self.convert_nice_time_to_seconds)
        return df


    def compute_and_label_time_diff(self, df: pd.DataFrame):
        """
        Computes the time differences between consecutive events and labels them

        The labeling is as follows:
          'No' if the time difference is between 0 and 10 seconds
          'Yes' otherwise (including negative differences beyond -10)
        """
        df = df.copy()
        df['time_diff'] = df['time_seconds'].diff().fillna(0)
        df['time_diff_label'] = df['time_diff'].apply(lambda x: 'No' if 0 <= x <= 10 else 'Yes')
        return df


    def get_phase_split_indices(self, df: pd.DataFrame, set_piece_events=None, other_phase_events=None):
        """
        Identifies indices where a new possession phase should start

        The function considers:
          Specific event types (e.g., set pieces or other phase-triggering events)
          Events where the time difference label is 'Yes'
        """
        if set_piece_events is None:
            set_piece_events = ['corner', 'freekick', 'throw_in', 'goalkick']
        if other_phase_events is None:
            other_phase_events = ['interception', 'goal', 'foul', 'offside', 'out',
                                  'keeper_save', 'keeper_claim', 'keeper_punch', 'keeper_pick_up']

        # Indices where the event type should trigger a phase split.
        split_indices_by_event = [
            idx for idx in df.index
            if df.loc[idx, 'type'] in set_piece_events or df.loc[idx, 'type'] in other_phase_events
        ]

        # Indices where the time difference indicates a break.
        split_indices_by_time = df.index[df['time_diff_label'] == 'Yes'].tolist()

        # Use only the time-split indices that are not already captured by event type.
        additional_split_indices = sorted(list(set(split_indices_by_time) - set(split_indices_by_event)))

        # Combine and sort all split indices.
        all_split_indices = sorted(split_indices_by_event + additional_split_indices)
        return all_split_indices


    def partition_indices(self, all_indices: list[int], split_indices: list[int]):
        """
        Partitions a list of indices based on split indices
        """
        return list(self.partition(all_indices, split_indices))


    def remove_preceding_out_indices(self, df: pd.DataFrame, index_partitions: list[list[int]]):
        """
        Removes indices from partitions that immediately precede an 'out' event
        """
        team_indices = df.index.tolist()
        preceding_out_indices = []

        # Identify indices preceding an 'out' event.
        for i in range(1, len(team_indices) - 1):
            current_idx = team_indices[i]
            if df.loc[current_idx, "type"] == 'out':
                preceding_out_indices.append(team_indices[i - 1])

        # Remove the identified indices from each partition.
        for phase in index_partitions:
            phase[:] = [idx for idx in phase if idx not in preceding_out_indices]

        return index_partitions


    def group_and_filter_phases(self, index_partitions: list[list[int]]):
        """
        Groups consecutive indices into phases and filters out phases that are too short
        Phases with 2 or fewer events are removed
        """
        valid_phases = []
        for phase_indices in index_partitions:
            groups = list(self._group_me(phase_indices))
            for group in groups:
                if len(group) > 2:
                    valid_phases.append(group)
        return valid_phases


    def map_indices_to_event_details(self, phases: list[list[int]], actions: pd.DataFrame):
        """
        Maps each index in a phase to its corresponding event details
        """
        event_details = actions.transpose().to_dict()
        phase_events = []
        for phase in phases:
            phase_events.append([event_details[idx] for idx in phase])
        return phase_events


    def split_possession_phases(self, actions: pd.DataFrame, team_name_mapping: dict[int, str]):
        """
        Splits possession sequences for each team into distinct phases.

        For each team, the function:
          1. Converts 'nice_time' strings to seconds
          2. Computes and labels time differences between events
          3. Identifies splits based on event types and time gaps
          4. Partitions and cleans the indices to form phases
          5. Groups and filters valid phases
          6. Maps indices back to event details
        """
        all_team_phases = []

        # Process each team's actions individually.
        for team_id in actions["team_id"].unique():

            club_name = team_name_mapping[team_id]

            # Filter actions for the current team.
            team_actions = actions[actions["team_id"] == team_id].copy()

            team_actions = self.add_time_seconds_column(team_actions)

            team_actions = self.compute_and_label_time_diff(team_actions)

            # Get all indices for the current team's actions.
            all_indices = team_actions.index.tolist()

            # Determine split indices based on event type and time differences.
            split_indices = self.get_phase_split_indices(team_actions)

            # Partition the indices into potential phases.
            index_partitions = self.partition_indices(all_indices, split_indices)

            # Remove actions that precede an 'out' event.
            index_partitions = self.remove_preceding_out_indices(team_actions, index_partitions)

            # Group consecutive indices and filter out phases with 2 or fewer events.
            valid_phases = self.group_and_filter_phases(index_partitions)

            # Map each phase's indices back to the corresponding event details.
            team_phase_events = self.map_indices_to_event_details(valid_phases, actions)

            all_team_phases.append([club_name, team_phase_events])

        return sum(all_team_phases, [])
    

class FilterPhases:
    # List of event types that should be removed.
    unwanted_event_types = [
        'foul', 'keeper_save', 'keeper_claim', 'keeper_punch', 'keeper_pick_up',
        'non_action', 'out', 'bad_touch', 'offside', 'receival', 'interception', 'goal']
    
    # Define event types that indicate a sequence should be removed if it starts with them.
    set_piece_events = ['throw_in', 'goalkick', 'yellow_card', 'red_card', 'freekick', 'corner']

    def __init__(self, phase):
        self.phase = phase

    def remove_unwanted_actions(self):
        """
        Removes unwanted actions from each sequence within a phase
        Unwanted actions include types such as fouls, keeper actions, non-actions, etc.
        """
        # Filter each sequence to remove events with types in unwanted_event_types.
        self.phase = [
            [event for event in sequence if event["type"] not in self.unwanted_event_types]
            for sequence in self.phase
        ]


    def filter_invalid_sequences(self):
        """
        Filters out invalid sequences from the phase based on several criteria:

        1. Removes entire sequences that start with a set piece event
        2. Removes the first 'dribble' action if it is the first event in a sequence
        3. Removes entire sequences that contain an 'owngoal' event
        """

        # Remove sequences that start with a set piece event.
        filtered_sequences = [
            sequence for sequence in self.phase
            if sequence and sequence[0]['type'] not in self.set_piece_events
        ]

        # Remove the first 'dribble' action if it is at the beginning of a sequence.
        for sequence in filtered_sequences:
            if sequence and sequence[0]['type'] == 'dribble':
                sequence.pop(0)

        # Remove sequences that contain an 'owngoal' event.
        valid_sequences = [
            sequence for sequence in filtered_sequences
            if not any(event['type'] == 'owngoal' for event in sequence)
        ]

        self.phase = valid_sequences

    def filter(self):
        self.remove_unwanted_actions()
        self.filter_invalid_sequences()
        return self.phase
    
def MakeMovementChains(phase: list[list[dict]]) -> list[list[dict]]:
    movement_chains = []
    for seq in phase:
        n = len(seq)
        I = 0, *(i for i in range(1, n) if seq[i]['player'] != seq[i-1]['player']), n
        chunks = [seq[i:j] for i, j in zip(I, I[4:])]
        movement_chains.append(chunks)
    movement_chains = sum(movement_chains, [])
    return movement_chains


def nice_time_to_seconds(t: str) -> int:
    m, s = t.replace("s", "").split("m")
    return int(m) * 60 + int(s)

def split_sequences_on_time_gaps(phases: list[dict], gap=10) -> list[dict]:
    """
    phases: list of sequences (each sequence = list of event dicts)
    Returns new list of sequences, where sequences are split when time gap >= 10 seconds
    """
    out = []
    for seq in phases:
        split_idx = []
        for j in range(len(seq) - 1):
            t_next = nice_time_to_seconds(seq[j + 1]["nice_time"])
            t_curr = nice_time_to_seconds(seq[j]["nice_time"])
            if t_next - t_curr >= gap:
                split_idx.append(j + 1)

        chunks = [list(x) for x in np.split(np.array(seq, dtype=object), split_idx) if len(x) > 0]
        out.extend(chunks)

    return out