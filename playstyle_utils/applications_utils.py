import numpy as np
import matplotlib.pyplot as plt

def standardized(metric_score: list[list[str, float]], teams: list) ->list[list[str, float]]:
    scores = np.array([score for _, score in metric_score])

    mu = scores.mean()
    sigma = scores.std(ddof=1)

    standardized_score = (scores - mu) / sigma

    standardized_team_score = [
    [club, z]
    for club, z in zip(teams, standardized_score)]

    return standardized_team_score



def aitchison_mean(list_of_lists):
    """
    Aitchison (compositional) mean for a list of compositions
    """
    X = np.asarray(list_of_lists, dtype=float)          
                           
    X = X / X.sum(axis=1, keepdims=True)               

    logX = np.log(X)
    clrX = logX - logX.mean(axis=1, keepdims=True)      

    mean_clr = clrX.mean(axis=0)                      
    mu = np.exp(mean_clr)                             
    return mu / mu.sum()

def home_vs_away(team, df_matches, topic_distributions_data):
    """
    Determines the play style of a team when playing at home vs away
    """

    home_game_id = set(df_matches.loc[df_matches["home_team_name"] == team, "game_id"])
    away_game_id = set(df_matches.loc[df_matches["away_team_name"] == team, "game_id"])

    home_styles = []
    away_styles = []

    for key, vec in topic_distributions_data.items():
        match_id, team_name = key.split("_", 1)
        match_id = int(match_id)
        
        if team_name != team:
            continue

        if match_id in home_game_id:
            home_styles.append(vec)
        elif match_id in away_game_id:
            away_styles.append(vec)

    out = {}
    out["home style"] = aitchison_mean(home_styles) 
    out["away style"] = aitchison_mean(away_styles) 

    return out


def split_matches(team, date, df_matches, topic_distributions_data):
    """
    Determines the play style of a team from matches before and after the given date
    """
    style_summary = {}
    df_before = df_matches.loc[((df_matches["home_team_name"] == team) | (df_matches["away_team_name"] == team)) & (df_matches["game_date"] < date)]
    df_after = df_matches.loc[((df_matches["home_team_name"] == team) | (df_matches["away_team_name"] == team)) & (df_matches["game_date"] > date)]

    for df in [df_before, df_after]:
        play_style = []
        for id in df["game_id"]:
            for match_key in topic_distributions_data.keys():
                match_key_id, match_key_team = match_key.split("_")[0], match_key.split("_")[1]
                if str(id) == match_key_id and team == match_key_team:
                    play_style.append(topic_distributions_data[match_key])
        if df.equals(df_before):
            style_summary[f"pre {date}"] = aitchison_mean(play_style)
        else:
            style_summary[f"post {date}"] = aitchison_mean(play_style)


    return style_summary


def make_show_plot(df_matches, topic_distributions_data):

    def show_plot(team: str, styles: dict, categories: list[str], date=None):
        serie_a_teams = {"Lazio", "Internazionale", "Roma", "Sassuolo", "Cagliari", "Atalanta", "Chievo", "Benevento", "Bologna", "Udinese", "Crotone", "Napoli", "Milan", "Fiorentina", "Sampdoria", "SPAL", "Torino", "Genoa", "Juventus", "Hellas Verona"}
        premier_league_teams = {"Burnley", "AFC Bournemouth", "Crystal Palace", "West Bromwich Albion", "Arsenal", "Huddersfield Town", "Brighton & Hove Albion", "Liverpool", "Watford", "Manchester United", "Newcastle United", "Chelsea", "Manchester City", "Southampton", "Swansea City", "Stoke City", "Leicester City", "Tottenham Hotspur", "Everton", "West Ham United"}
        la_liga_teams = {"Barcelona", "Real Sociedad", "Atlético Madrid", "Eibar", "Espanyol", "Athletic Club", "Valencia", "Deportivo La Coruña", "Real Madrid", "Villarreal", "Deportivo Alavés", "Sevilla", "Getafe", "Málaga", "Las Palmas", "Girona", "Real Betis", "Leganés", "Celta de Vigo", "Levante"}
        ligue1_teams = {"Caen", "PSG", "Dijon", "Angers", "Olympique Lyonnais", "Nice", "Olympique Marseille", "Amiens SC", "Bordeaux", "Metz", "Strasbourg", "Nantes", "Montpellier", "Rennes", "Saint-Étienne", "Lille", "Toulouse", "Guingamp", "Troyes", "Monaco"}
        bundesliga_teams = {"Bayern München", "Stuttgart", "Hoffenheim", "Borussia Dortmund", "Hertha BSC", "RB Leipzig", "Freiburg", "Augsburg", "Schalke 04", "Eintracht Frankfurt", "Hannover 96", "Bayer Leverkusen", "Borussia M'gladbach", "Hamburger SV", "Werder Bremen", "Mainz 05", "Wolfsburg", "Köln"}

        color_1 = "#515153C3"
        color_2 = "#9d0208"

        if team in serie_a_teams:
            league = "Serie A"
        elif team in ligue1_teams:
            league = "Ligue 1"
        elif team in premier_league_teams:
            league = "Premier League"
        elif team in la_liga_teams:
            league = "La Liga"
        elif team in bundesliga_teams:
            league = "Bundesliga"
        else:
            league = ""

        x = np.arange(len(categories))
        fig, ax = plt.subplots(figsize=(8, 5))

        if date is None:
            home_styles = styles["home style"]
            away_styles = styles["away style"]
            ax.plot(x, home_styles, color=color_1, marker="o", linewidth=2, label="Home")
            ax.plot(x, away_styles, color=color_2, marker="s", linestyle="--", linewidth=2, label="Away")
        else:
            pre_and_post_style = split_matches(team, date, df_matches, topic_distributions_data)
            pre_style = pre_and_post_style[f"pre {date}"]
            post_style = pre_and_post_style[f"post {date}"]
            ax.plot(x, pre_style, color=color_1, marker="o", linewidth=2, label=f"pre {date}")
            ax.plot(x, post_style, color=color_2, marker="s", linestyle="--", linewidth=2, label=f"post {date}")

        ax.set_ylabel("Proportion")
        ax.set_title(f"{team} - {league}")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    return show_plot


def plot_club_styles(club_style_distributions: dict, teams_to_plot: list[str],
                     categories: list[str], title: str, figsize=(8, 5)):
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=figsize)

    for team in teams_to_plot:
        if team not in club_style_distributions:
            continue
        mean = club_style_distributions[team]
        ax.plot(x, mean, label=team)

    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()