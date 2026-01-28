# Identifying playing styles in football using topic modeling

This project turns raw soccer event data into **team-level “play style” topic distributions** by:
1) converting match events → **SPADL** → **Atomic-SPADL**  
2) splitting actions into **possession phases** and **movement chains**  
3) converting movement chains to **trajectories** (x,y paths)  
4) filtering **noise trajectories**  
5) clustering trajectories using **DTW** + **medoids**  
6) representing each match/team as a “document” of trajectory-cluster IDs  
7) applying **LDA (Latent Dirichlet Allocation)** to learn **topics = play styles**  

The end result is an interpretable representation of how teams play, based on recurring movement patterns and their topic mixtures.

## What’s in this repo
```bash
.
├── data/ (not committed) raw / extracted data files
├── notebooks/
│   ├── 01_atomic_spadl.ipynb
│   ├── 02_phases_and_chains.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_LDA.ipynb
│   ├── algorithms.ipynb
│   └── applications.ipynb
├── playstyle/
├── playstyle_utils/
│   ├── __init__.py
│   ├── algorithm_utils.py
│   ├── applications_utils.py
│   ├── bezier_utils.py
│   ├── clustering.py
│   ├── compositional.py
│   ├── dtw.py
│   ├── noise.py
│   ├── phases.py
│   └── spadl_atomic.py
├── tests/
│   ├── test_composition.py
│   ├── test_dtw.py
│   ├── test_movement_chains.py
│   ├── test_noise.py
│   └── test_phases.py
└── README.md
```
## Quick start 
```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Run notebooks in order
### `notebooks/01_atomic_spadl.ipynb`
- Converts Wyscout events -> SPADL -> Atomic-SPADL
- Builds `match_events`

### `notebooks/02_phases_and_chains.ipynb`
- Splits each team’s actions into possession phases
- Filters sequences
- Builds movement chains
- Extracts trajectories
- Removes noise

### `notebooks/03_clustering.ipynb`
- Runs DTW + medoids clustering
- Assigns each trajectory to a medoid 
- Builds match/team “documents” of cluster medoids

### `notebooks/04_LDA.ipynb`
- Fits LDA

### `notebooks/05_applications.ipynb`
- Computes team play-style topic distributions 
- Compares team play-style uniqueness vs consistency for a given league
- Compares home vs away play style for a selected team
- Computes pre- vs post- date play-style changes for a selected team

## Reproducibility between notebooks

Notebook outputs (like `match_events`, `team_name_mapping`, etc.) are saved to disk so later notebooks can load them without rerunning everything.

## High level overview of the methods

### Possession phases
Match event data is split into phases, which are uninterrupted sequences of events where on team is in possession of the ball, using:
- set-pieces (e.g., corner, foul, out, etc.)
- team ball possession (i.e., a team losing possession of the ball starts the possession phase of their opponent)
- time gaps (**>= 10s**) between events

### Movement chains
Within each phase, sequences are split by player changes into movement chains (i.e.,phases that describe four consecutive player involvements).

### Trajectory clustering (DTW + medoids)
- Each movement chain becomes a 2D trajectory
- Distance between trajectories = DTW on (x, y)
- Clusters are represented by medoids
- Every trajectory is assigned the closest medoid

### LDA topics = play styles
After clustering, we treat each match/team as a document:
- **Words** = trajectory cluster IDs (e.g., medoid index)
- **Document** = all cluster IDs observed for that team in that match

Then we fit **LDA (Latent Dirichlet Allocation)**:
- **Topics** = combinations of trajectory clusters that co-occur frequently
- Interpreted as play styles
- Output per match/team: a topic probability vector (e.g., `[p(topic1), ..., p(topicK)]`)
- Output per topic: a distribution over cluster IDs (which can be inspected/visualized)

---

## Data

This project uses the public Wyscout spatio-temporal event dataset from:

Pappalardo, L., Cintia, P., Rossi, A. et al. *A public data set of spatio-temporal match events in soccer competitions.* **Scientific Data** 6, 236 (2019). https://doi.org/10.1038/s41597-019-0247-7
