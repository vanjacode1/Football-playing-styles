import socceraction
import socceraction.spadl.wyscout as wyscout 
import socceraction.atomic.spadl as atomicspadl

class EventToAtomic:
    def __init__(self, game_id: int, home_id: int, team_name_mapping: dict[int, str], player_name_mapping: dict[int, str], atomic_type_mapping: dict[int, str], wsl):
        self.game_id = game_id
        self.home_id = home_id
        self.team_name_mapping = team_name_mapping
        self.player_name_mapping = player_name_mapping
        self.atomic_type_mapping = atomic_type_mapping
        self.wsl = wsl
        
    def convert_to_atomic(self):
        """
        Convert a Wyscout match (events) into Atomic-SPADL actions, oriented left-to-right
        """
        #Raw event stream for the match
        df_events = self.wsl.events(self.game_id)

        #Convert Wyscout events to SPADL actions
        df_actions = wyscout.convert_to_actions(df_events, self.home_id)

        #Orient actions from left to right for every match
        df_actions = socceraction.spadl.play_left_to_right(df_actions, self.home_id)

        #Convert SPADL actions to Atomic SPADL
        df_atomic_actions = atomicspadl.convert_to_atomic(df_actions)

        return df_atomic_actions
    
    @staticmethod
    def _nice_time(row):
        """
        Convert the time in a game in a smart way to minutes and seconds "XmYs"
        """
        minute = int((row['period_id']>=2) * 45 + (row['period_id']>=3) * 15 + 
                     (row['period_id']==4) * 15 + row['time_seconds'] // 60)
        second = int(row['time_seconds'] % 60)
        return f'{minute}m{second}s'
    

    def complete_atomic_events(self):
        """
        Expand Atomic SPADL actions with start/end coordinates, readable type/team/player,
        and a "XmYs" timestamp
        """
        df_atomic_actions = self.convert_to_atomic()
        #df_atomic_actions = df_atomic_actions.copy()
        df_atomic_actions["start_x"] = df_atomic_actions.x
        df_atomic_actions["start_y"] = df_atomic_actions.y
        df_atomic_actions["end_x"] = df_atomic_actions.x + df_atomic_actions.dx
        df_atomic_actions["end_y"] = df_atomic_actions.y + df_atomic_actions.dy

        df_atomic_actions["type"] = df_atomic_actions["type_id"].map(self.atomic_type_mapping)
        df_atomic_actions["nice_time"] = df_atomic_actions.apply(self._nice_time, axis=1)
        df_atomic_actions["team"] = df_atomic_actions['team_id'].map(self.team_name_mapping)
        df_atomic_actions["player"] = df_atomic_actions["player_id"].map(self.player_name_mapping)

        return df_atomic_actions[["nice_time", "player", 'start_x', 'start_y', 'end_x', 'end_y', "type", "team", "team_id"]]