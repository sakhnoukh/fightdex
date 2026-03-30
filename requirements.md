# PokéCoach — Final Project Requirements Document

> **Course:** Recommendation Systems — Group Final Project  
> **Format:** Jupyter Notebook + CEO Presentation  
> **Concept:** Context-aware hybrid recommender that finds the optimal next Pokémon for a VGC team, evaluated against live battle simulations

---

## 1. Project Vision

### What we are building

PokéCoach answers a single, well-defined question:

> *Given the Pokémon I have already chosen (0–5) and any opponent Pokémon I can see (0–6), what is the best next Pokémon to add to my team?*

This is a pure optimisation problem — no real users, no personal histories. The system finds the objectively strongest candidate based on three simultaneous signals, then validates its recommendations by simulating actual battles.

### Why this is a recommendation problem

- **Items** = Pokémon (from the legal regulation set pool)
- **Context A** = own partial team (0–5 already chosen Pokémon)
- **Context B** = opponent's known Pokémon (0–6 revealed)
- **Score** = weighted combination of synergy, counter strength, and viability
- **Ground truth** = simulated win rate from the poke-env battle simulator

### The scoring formula

```
score(c) = α · synergy(c) + β · counter(c) + γ · viability(c)

where α + β + γ = 1
```

Weights are **context-dependent**:

| Situation | α (synergy) | β (counter) | γ (viability) |
|---|---|---|---|
| No team, no opponent known | 0.0 | 0.0 | 1.0 |
| Partial team, no opponent | 0.6 | 0.0 | 0.4 |
| Partial team, partial opponent | 0.4 | 0.3 | 0.3 |
| Full team of 5, full opponent known | 0.3 | 0.5 | 0.2 |

This graceful degradation is the core of the context-aware design and must be explicitly discussed in the methodology.

### CEO pitch (one sentence)

> "We built a system that recommends the single best Pokémon for any competitive team in any matchup scenario — and we can prove it works because the teams it builds win measurably more simulated battles than teams built by any baseline approach."

---

## 2. Datasets

### D1 — Smogon VGC Usage Statistics

**What it is:** Monthly aggregated usage data from Pokémon Showdown ladder battles, stratified by Elo rating. The most important file type for this project.

**What it contains:**
- Overall usage % per Pokémon in the format
- **Teammate co-occurrence table**: for each Pokémon, the % of its teams that also carry every other Pokémon — this is your item-item co-occurrence matrix for collaborative filtering
- Checks & counters data
- Moveset frequency tables: most common moves (slot 1–4), items, abilities, spreads — used to build canonical team definitions for simulation

**URL pattern (confirmed working):**
```
https://www.smogon.com/stats/{YYYY-MM}/{format}-{elo}.txt
```

**Relevant format strings for Gen 9 VGC:**
```
gen9vgc2024regg     # Regulation G — May–Aug 2024, Jan–Apr 2025 (use as TRAIN)
gen9vgc2024regh     # Regulation H — Sep 2024–Jan 2025 (use as TEST)
gen9vgc2024regf     # Regulation F — Jan–Apr 2024 (optional extra train)
```

**Download commands:**

```bash
mkdir -p data/smogon && cd data/smogon

# Regulation G — download all available months (train set)
for month in 2024-05 2024-06 2024-07 2024-08; do
  curl -sO "https://www.smogon.com/stats/${month}/gen9vgc2024regg-1630.txt"
done

# Regulation H — download all available months (test set)
for month in 2024-09 2024-10 2024-11 2024-12; do
  curl -sO "https://www.smogon.com/stats/${month}/gen9vgc2024regh-1630.txt"
done

# Verify files downloaded
ls -lh *.txt
```

**Python loading skeleton:**

```python
import requests
import re
import pandas as pd

def parse_smogon_usage(url: str) -> pd.DataFrame:
    """Parse the usage table section from a Smogon stats .txt file."""
    raw = requests.get(url).text
    lines = raw.split("\n")
    
    records = []
    in_usage = False
    for line in lines:
        if "| Rank | Pokemon" in line:
            in_usage = True
            continue
        if in_usage and line.startswith(" +"):
            break
        if in_usage and line.startswith(" |"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[1].isdigit():
                records.append({
                    "rank": int(parts[1]),
                    "pokemon": parts[2],
                    "usage_pct": float(parts[3].replace("%", ""))
                })
    return pd.DataFrame(records)


def parse_smogon_teammates(url: str) -> dict:
    """
    Parse the Teammates section for every Pokémon.
    Returns dict: {pokemon_name: {teammate_name: co_occurrence_pct}}
    This forms the item-item co-occurrence matrix for collaborative filtering.
    """
    raw = requests.get(url).text
    result = {}
    current_pokemon = None
    in_teammates = False

    for line in raw.split("\n"):
        # Detect new Pokémon block
        match = re.match(r"\s*\|\s*([A-Za-z\-\. ]+)\s*\|\s*$", line)
        if match and len(line.strip()) < 40:
            current_pokemon = match.group(1).strip()
            in_teammates = False

        if current_pokemon and "Teammates" in line:
            in_teammates = True
            result[current_pokemon] = {}
            continue

        if in_teammates and "|" in line and "%" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    teammate = parts[1].strip()
                    pct = float(parts[2].replace("%", "").strip())
                    result[current_pokemon][teammate] = pct
                except ValueError:
                    pass

        # Stop at next section
        if in_teammates and ("Checks and Counters" in line or "Spreads" in line):
            in_teammates = False

    return result
```

---

### D2 — PokeAPI CSV Dump (GitHub)

**What it is:** The complete relational Pokémon database exported as flat CSV files. The `type_efficacy.csv` is the critical file — it is the 18×18 type effectiveness multiplier table that powers the counter signal.

**Key files:**

| File | Contents | Used for |
|---|---|---|
| `pokemon.csv` | Pokémon IDs and names | Joining tables |
| `pokemon_stats.csv` | HP/Atk/Def/SpA/SpD/Spe per Pokémon | Content features |
| `pokemon_types.csv` | Type assignments (handles dual-type) | Content features, counter calc |
| `type_efficacy.csv` | Full 18×18 damage multiplier table | Counter signal |
| `pokemon_species.csv` | Generation, legendary flag | Filtering legal pool |
| `abilities.csv` + `pokemon_abilities.csv` | Ability assignments | Content features |

**Download commands:**

```bash
mkdir -p data/pokeapi && cd data/pokeapi

BASE="https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv"

for file in pokemon.csv pokemon_stats.csv pokemon_types.csv type_efficacy.csv \
            pokemon_species.csv abilities.csv pokemon_abilities.csv \
            types.csv stat_names.csv; do
  curl -sO "${BASE}/${file}"
  echo "Downloaded ${file}"
done
```

**Python loading:**

```python
import pandas as pd

BASE = "data/pokeapi"

pokemon        = pd.read_csv(f"{BASE}/pokemon.csv")
pokemon_stats  = pd.read_csv(f"{BASE}/pokemon_stats.csv")
pokemon_types  = pd.read_csv(f"{BASE}/pokemon_types.csv")
type_efficacy  = pd.read_csv(f"{BASE}/type_efficacy.csv")
pokemon_species = pd.read_csv(f"{BASE}/pokemon_species.csv")
abilities      = pd.read_csv(f"{BASE}/abilities.csv")
```

**Key detail for `type_efficacy.csv`:**
The `damage_factor` column is the multiplier × 100. So 200 = 2× (super effective), 50 = 0.5× (not very effective), 0 = immune. To get the actual multiplier: `multiplier = damage_factor / 100`.

---

### D3 — poke-env Battle Simulator (self-generated dataset)

**What it is:** Not a downloaded dataset — a dataset you generate yourself by running simulated battles. This is your ground-truth signal for evaluating recommendations.

**What you generate:** A CSV with rows like:
```
team_composition, win_rate_vs_heuristics, n_battles, regulation_set
[Tornadus,Rillaboom,Incineroar,Urshifu,Amoonguss,Flutter Mane], 0.72, 50, reg_g
```

**Setup (one-time, ~30 minutes):**

```bash
# Step 1: Install Node.js (required for Showdown server)
# macOS:  brew install node
# Ubuntu: sudo apt install nodejs npm

# Step 2: Clone and start the Showdown server locally
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
# Server now running at localhost:8000

# Step 3: Install poke-env (in a new terminal)
pip install poke-env
```

**Verify the setup works:**

```python
import asyncio
from poke_env import RandomPlayer

async def test():
    p1 = RandomPlayer(battle_format="gen9vgc2024regg")
    p2 = RandomPlayer(battle_format="gen9vgc2024regg")
    await p1.battle_against(p2, n_battles=1)
    print(f"Won: {p1.n_won_battles} / 1")

asyncio.run(test())
```

**Speed benchmarks:**
- RandomPlayer vs RandomPlayer: ~10 battles/sec
- SimpleHeuristicsPlayer vs SimpleHeuristicsPlayer (VGC doubles): ~2–4 battles/sec
- 50 battles for one team: ~15–25 seconds
- 200 teams × 50 battles: ~25–40 minutes (parallelisable)

---

## 3. Data Pipeline

### Step 1 — Build the legal Pokémon pool

Filter `pokemon_species.csv` to only Pokémon available in Regulation G (generation ≤ 9, excluding restricted legendaries). Cross-reference with the Smogon usage file to get only Pokémon that actually appear in competitive play (usage > 0.5%).

```python
def build_legal_pool(usage_df: pd.DataFrame, min_usage: float = 0.5) -> list:
    """Return list of Pokémon names viable in the current regulation."""
    return usage_df[usage_df["usage_pct"] >= min_usage]["pokemon"].tolist()
```

### Step 2 — Build the co-occurrence matrix

From the Smogon teammate tables, build a symmetric Pokémon × Pokémon matrix where each cell contains the percentage of teams running Pokémon A that also run Pokémon B. This is the input to all collaborative filtering models.

```python
def build_cooccurrence_matrix(teammate_dict: dict, legal_pool: list) -> pd.DataFrame:
    """
    Build a square Pokémon × Pokémon co-occurrence matrix.
    Rows/cols = Pokémon in legal pool.
    Values = co-occurrence percentage (0–100).
    """
    df = pd.DataFrame(0.0, index=legal_pool, columns=legal_pool)
    for pokemon, teammates in teammate_dict.items():
        if pokemon not in legal_pool:
            continue
        for teammate, pct in teammates.items():
            if teammate in legal_pool:
                df.loc[pokemon, teammate] = pct
                df.loc[teammate, pokemon] = max(df.loc[teammate, pokemon], pct)
    return df
```

### Step 3 — Build content feature vectors

For each Pokémon, create a numerical feature vector combining base stats, type (one-hot encoded across 18 types), and role indicators.

```python
def build_content_features(pokemon_df, stats_df, types_df, legal_pool) -> pd.DataFrame:
    """
    Returns DataFrame with one row per Pokémon and columns:
    hp, attack, defense, sp_atk, sp_def, speed,  (normalised 0–1)
    type_normal, type_fire, ..., type_fairy,       (one-hot, 18 cols)
    is_fast, is_bulky, is_attacker, is_support     (derived role tags)
    """
    # ... implementation
```

### Step 4 — Build the type effectiveness counter matrix

From `type_efficacy.csv`, compute for each pair of (attacker_types, defender_types) the combined damage multiplier. This powers the counter signal.

```python
def build_type_chart(type_efficacy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns 18×18 DataFrame.
    Rows = attacking type, Cols = defending type.
    Values = damage multiplier (0, 0.25, 0.5, 1, 2, 4).
    """
    chart = type_efficacy_df.pivot(
        index="damage_type_id",
        columns="target_type_id",
        values="damage_factor"
    ) / 100.0
    return chart


def counter_score(candidate: str, opponent_team: list,
                  pokemon_types: dict, type_chart: pd.DataFrame) -> float:
    """
    Score how well `candidate` counters the known opponent Pokémon.
    Returns float in [0, 1].
    Higher = better type advantage over the opponent team.
    """
    if not opponent_team:
        return 0.5  # neutral when no opponent info
    
    scores = []
    cand_types = pokemon_types[candidate]
    
    for opp in opponent_team:
        opp_types = pokemon_types[opp]
        # Offensive: how effectively can candidate hit opponent?
        offensive = max(
            type_chart.loc[t_atk, t_def]
            for t_atk in cand_types
            for t_def in opp_types
        )
        # Defensive: how much does opponent threaten candidate?
        defensive = max(
            type_chart.loc[t_atk, t_def]
            for t_atk in opp_types
            for t_def in cand_types
        )
        # Score = good offensive matchup AND surviving hits
        scores.append(offensive / (defensive + 1e-6))
    
    raw = sum(scores) / len(scores)
    return min(raw / 4.0, 1.0)  # normalise to [0, 1]
```

### Step 5 — Build canonical team pastes for simulation

For each Pokémon in the legal pool, extract the most common moveset from the Smogon stats file and generate a valid Showdown paste string.

```python
def build_canonical_paste(pokemon_name: str, moveset_data: dict) -> str:
    """
    Generate a Showdown-format team paste string for a single Pokémon.
    moveset_data comes from parsing the Smogon stats moveset section.
    
    Returns string like:
    Tornadus @ Covert Cloak
    Ability: Prankster
    Level: 50
    EVs: 252 HP / 4 Def / 252 Spe
    Timid Nature
    - Tailwind
    - Bleakwind Storm
    - Protect
    - Taunt
    """
    m = moveset_data[pokemon_name]
    moves = "\n".join(f"- {move}" for move in m["top_moves"][:4])
    return (
        f"{pokemon_name} @ {m['top_item']}\n"
        f"Ability: {m['top_ability']}\n"
        f"Level: 50\n"
        f"EVs: {m['top_spread']}\n"
        f"{m['top_nature']} Nature\n"
        f"{moves}\n"
    )


def build_team_paste(team: list, canonical_movesets: dict) -> str:
    """Concatenate 6 Pokémon pastes into a full team string."""
    return "\n\n".join(
        build_canonical_paste(p, canonical_movesets) for p in team
    )
```

### Step 6 — Generate the simulation dataset

Run simulated battles for a sample of team configurations. Record win rates as ground truth labels.

```python
import asyncio
from poke_env import RandomPlayer
from poke_env.player import SimpleHeuristicsPlayer

async def evaluate_team(team_paste: str, n_battles: int = 50) -> float:
    """
    Simulate n_battles with the given team against SimpleHeuristicsPlayer.
    Returns win rate as float in [0, 1].
    """
    class TeamPlayer(SimpleHeuristicsPlayer):
        def teambuilder(self):
            return team_paste  # use our canonical paste

    player = TeamPlayer(battle_format="gen9vgc2024regg")
    opponent = SimpleHeuristicsPlayer(battle_format="gen9vgc2024regg")
    
    await player.battle_against(opponent, n_battles=n_battles)
    return player.n_won_battles / n_battles


async def generate_simulation_dataset(
    team_configs: list,          # list of (team_name, [6 pokemon names])
    canonical_movesets: dict,
    n_battles: int = 50,
    output_path: str = "data/simulation_results.csv"
) -> pd.DataFrame:
    """
    Run simulations for all team configs and save results.
    Runs concurrently for speed.
    """
    results = []
    
    async def run_one(name, team):
        paste = build_team_paste(team, canonical_movesets)
        win_rate = await evaluate_team(paste, n_battles)
        results.append({"team_name": name, "team": team, "win_rate": win_rate})
        print(f"  {name}: {win_rate:.2f} ({n_battles} battles)")
    
    tasks = [run_one(name, team) for name, team in team_configs]
    await asyncio.gather(*tasks)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df
```

---

## 4. Recommendation Models

### Model 0 — Random baseline

```python
import random

def recommend_random(own_team: list, legal_pool: list, top_n: int = 5) -> list:
    candidates = [p for p in legal_pool if p not in own_team]
    return random.sample(candidates, min(top_n, len(candidates)))
```

### Model 1 — Popular baseline

```python
def recommend_popular(own_team: list, usage_df: pd.DataFrame, top_n: int = 5) -> list:
    candidates = usage_df[~usage_df["pokemon"].isin(own_team)]
    return candidates.nlargest(top_n, "usage_pct")["pokemon"].tolist()
```

### Model 2 — Memory-based CF (KNN on co-occurrence)

Find the most similar Pokémon to each member of the own team, then rank candidates by how frequently they co-occur with the team as a whole.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_knn(
    own_team: list,
    cooccurrence_matrix: pd.DataFrame,
    top_n: int = 5,
    k: int = 10
) -> list:
    """
    Item-item KNN: for each Pokémon in own_team, find k most similar
    Pokémon by cosine similarity on co-occurrence vectors.
    Score candidates by summed similarity across team members.
    """
    candidates = [p for p in cooccurrence_matrix.index if p not in own_team]
    
    if not own_team:
        # Cold start: return most connected items
        scores = cooccurrence_matrix[candidates].sum()
        return scores.nlargest(top_n).index.tolist()
    
    scores = pd.Series(0.0, index=candidates)
    
    for member in own_team:
        if member not in cooccurrence_matrix.index:
            continue
        member_vec = cooccurrence_matrix.loc[member].values.reshape(1, -1)
        candidate_vecs = cooccurrence_matrix.loc[candidates].values
        sims = cosine_similarity(member_vec, candidate_vecs)[0]
        scores += pd.Series(sims, index=candidates)
    
    return scores.nlargest(top_n).index.tolist()
```

### Model 3 — Model-based CF (SVD / NMF)

Factorise the co-occurrence matrix into latent "archetype" dimensions. Each latent factor corresponds to something like "trick room team", "sun team", "hyper-offense team", etc.

```python
from sklearn.decomposition import TruncatedSVD, NMF

def fit_svd(cooccurrence_matrix: pd.DataFrame, n_components: int = 20):
    """Fit SVD on co-occurrence matrix. Returns model and Pokémon embeddings."""
    model = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = model.fit_transform(cooccurrence_matrix.values)
    return model, pd.DataFrame(embeddings, index=cooccurrence_matrix.index)


def recommend_svd(
    own_team: list,
    embeddings: pd.DataFrame,
    top_n: int = 5
) -> list:
    """
    Reconstruct the team's latent profile by averaging member embeddings.
    Recommend candidates with highest cosine similarity to that profile.
    """
    candidates = [p for p in embeddings.index if p not in own_team]
    
    if not own_team:
        # Fallback: return candidates with highest overall embedding norm
        norms = np.linalg.norm(embeddings.loc[candidates].values, axis=1)
        return embeddings.loc[candidates].iloc[np.argsort(-norms)[:top_n]].index.tolist()
    
    team_vecs = embeddings.loc[[p for p in own_team if p in embeddings.index]]
    team_profile = team_vecs.mean(axis=0).values.reshape(1, -1)
    
    candidate_vecs = embeddings.loc[candidates].values
    sims = cosine_similarity(team_profile, candidate_vecs)[0]
    
    return pd.Series(sims, index=candidates).nlargest(top_n).index.tolist()
```

### Model 4 — Content-based recommender

Recommend candidates that fill gaps in type coverage or speed tier relative to the existing team.

```python
def recommend_content(
    own_team: list,
    content_features: pd.DataFrame,
    top_n: int = 5
) -> list:
    """
    Score candidates by how much they add to the team's type diversity
    and stat profile. Penalise type overlap with existing members.
    """
    candidates = [p for p in content_features.index if p not in own_team]
    
    if not own_team:
        # No team: return highest viability (BST-based)
        bst = content_features.loc[candidates, ["hp","attack","defense",
                                                 "sp_atk","sp_def","speed"]].sum(axis=1)
        return bst.nlargest(top_n).index.tolist()
    
    team_types = content_features.loc[
        [p for p in own_team if p in content_features.index],
        [c for c in content_features.columns if c.startswith("type_")]
    ].max(axis=0)  # which types are already covered
    
    scores = {}
    for candidate in candidates:
        cand_types = content_features.loc[
            candidate, [c for c in content_features.columns if c.startswith("type_")]
        ]
        # Novelty: how many new type coverages does this add?
        new_coverage = ((cand_types > 0) & (team_types == 0)).sum()
        # Stat contribution: normalised BST
        bst_score = content_features.loc[
            candidate, ["hp","attack","defense","sp_atk","sp_def","speed"]
        ].sum() / 600.0
        
        scores[candidate] = 0.6 * new_coverage / 18.0 + 0.4 * bst_score
    
    return pd.Series(scores).nlargest(top_n).index.tolist()
```

### Model 5 — NLP content-based (TF-IDF on Smogon analyses)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_embeddings(analyses: dict) -> tuple:
    """
    analyses: dict of {pokemon_name: analysis_text}
    Returns (vectorizer, tfidf_matrix, index_to_name)
    """
    names = list(analyses.keys())
    texts = [analyses[n] for n in names]
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, names


def recommend_tfidf(
    own_team: list,
    tfidf_matrix,
    index_to_name: list,
    top_n: int = 5
) -> list:
    """Find candidates most similar in role to the team's existing members."""
    name_to_idx = {n: i for i, n in enumerate(index_to_name)}
    candidates = [n for n in index_to_name if n not in own_team]
    candidate_idxs = [name_to_idx[n] for n in candidates]
    
    if not own_team:
        return candidates[:top_n]
    
    team_idxs = [name_to_idx[p] for p in own_team if p in name_to_idx]
    team_matrix = tfidf_matrix[team_idxs]
    team_profile = team_matrix.mean(axis=0)
    
    cand_matrix = tfidf_matrix[candidate_idxs]
    sims = cosine_similarity(team_profile, cand_matrix)[0]
    
    top_idxs = np.argsort(-sims)[:top_n]
    return [candidates[i] for i in top_idxs]
```

### Model 6 — Hybrid recommender (the main model)

```python
def recommend_hybrid(
    own_team: list,
    opponent_team: list,
    legal_pool: list,
    usage_df: pd.DataFrame,
    cooccurrence_matrix: pd.DataFrame,
    svd_embeddings: pd.DataFrame,
    content_features: pd.DataFrame,
    pokemon_types: dict,
    type_chart: pd.DataFrame,
    top_n: int = 5,
    diversity_penalty: float = 0.2
) -> pd.DataFrame:
    """
    Full hybrid recommender combining all three signals.
    Returns ranked DataFrame with score breakdown.
    """
    # --- Compute context-dependent weights ---
    n_own = len(own_team)
    n_opp = len(opponent_team)
    
    if n_own == 0:
        alpha, beta, gamma = 0.0, 0.0, 1.0
    elif n_opp == 0:
        alpha, beta, gamma = 0.6, 0.0, 0.4
    elif n_opp <= 2:
        alpha, beta, gamma = 0.4, 0.3, 0.3
    else:
        alpha, beta, gamma = 0.3, 0.5, 0.2

    candidates = [p for p in legal_pool if p not in own_team]
    
    # --- Signal A: Synergy (SVD-based CF) ---
    if n_own > 0 and not svd_embeddings.empty:
        team_vecs = svd_embeddings.loc[[p for p in own_team if p in svd_embeddings.index]]
        if not team_vecs.empty:
            profile = team_vecs.mean(axis=0).values.reshape(1, -1)
            cand_vecs = svd_embeddings.loc[
                [p for p in candidates if p in svd_embeddings.index]
            ].values
            valid_candidates = [p for p in candidates if p in svd_embeddings.index]
            synergy_sims = cosine_similarity(profile, cand_vecs)[0]
            synergy_scores = pd.Series(synergy_sims, index=valid_candidates)
        else:
            synergy_scores = pd.Series(0.5, index=candidates)
    else:
        synergy_scores = pd.Series(0.5, index=candidates)
    
    # --- Signal B: Counter (type effectiveness) ---
    counter_scores = pd.Series({
        p: counter_score(p, opponent_team, pokemon_types, type_chart)
        for p in candidates
    })
    
    # --- Signal C: Viability (usage % normalised) ---
    max_usage = usage_df["usage_pct"].max()
    usage_lookup = usage_df.set_index("pokemon")["usage_pct"] / max_usage
    viability_scores = pd.Series({
        p: usage_lookup.get(p, 0.1) for p in candidates
    })
    
    # --- Align all series ---
    all_candidates = list(set(synergy_scores.index)
                          & set(counter_scores.index)
                          & set(viability_scores.index))
    
    final_scores = (
        alpha * synergy_scores[all_candidates] +
        beta  * counter_scores[all_candidates] +
        gamma * viability_scores[all_candidates]
    )
    
    # --- Diversity re-rank: penalise type overlap with own team ---
    if own_team and diversity_penalty > 0:
        own_type_set = set()
        for p in own_team:
            own_type_set.update(pokemon_types.get(p, []))
        
        for p in all_candidates:
            cand_types = set(pokemon_types.get(p, []))
            overlap = len(cand_types & own_type_set) / max(len(cand_types), 1)
            final_scores[p] *= (1 - diversity_penalty * overlap)
    
    result = pd.DataFrame({
        "pokemon": all_candidates,
        "score": final_scores[all_candidates].values,
        "synergy": synergy_scores[all_candidates].values,
        "counter": counter_scores[all_candidates].values,
        "viability": viability_scores[all_candidates].values,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }).sort_values("score", ascending=False).head(top_n)
    
    return result
```

### Model 7 — Bandit-augmented hybrid

```python
import numpy as np

class UCB1Recommender:
    """
    UCB1 bandit over the hybrid recommender's top candidates.
    Explores uncertain counter picks when opponent is partially known.
    Exploitation = synergy score. Exploration = uncertainty from partial opponent info.
    """
    def __init__(self, c: float = 1.41):
        self.c = c           # exploration coefficient
        self.counts = {}     # {pokemon: times recommended}
        self.rewards = {}    # {pokemon: cumulative win rate}
    
    def ucb_score(self, pokemon: str, base_score: float, total_rounds: int) -> float:
        n = self.counts.get(pokemon, 0)
        if n == 0:
            return float("inf")  # always try unexplored candidates
        exploitation = self.rewards.get(pokemon, 0) / n
        exploration = self.c * np.sqrt(np.log(total_rounds + 1) / n)
        return exploitation + exploration + 0.1 * base_score
    
    def recommend(self, hybrid_candidates: pd.DataFrame, top_n: int = 5) -> list:
        total = sum(self.counts.values()) + 1
        scored = {
            row["pokemon"]: self.ucb_score(row["pokemon"], row["score"], total)
            for _, row in hybrid_candidates.iterrows()
        }
        return sorted(scored, key=scored.get, reverse=True)[:top_n]
    
    def update(self, pokemon: str, won: bool):
        self.counts[pokemon] = self.counts.get(pokemon, 0) + 1
        self.rewards[pokemon] = self.rewards.get(pokemon, 0.0) + float(won)
```

---

## 5. Evaluation Framework

### 5.1 CVTT split

**Train:** Regulation G data (Smogon stats from May–August 2024)  
**Test:** Regulation H data (Smogon stats from September–December 2024)

This is a genuine temporal split — the Regulation H meta introduced new Pokémon and shifted usage patterns significantly from Regulation G. Training on G and testing on H prevents meta-leakage and is the correct real-world evaluation protocol.

### 5.2 Offline metrics (from Smogon test data)

Use the **team reconstruction task**: take a known high-usage team composition from Regulation H, mask one Pokémon, and evaluate whether the model recovers it.

```python
def evaluate_reconstruction(
    model_fn,           # callable: (own_team, opponent_team) -> ranked list
    test_teams: list,   # list of 6-Pokémon teams from Reg H
    top_n: int = 5
) -> dict:
    """
    For each test team:
      - Mask one Pokémon (leave-one-out)
      - Ask the model to recommend top_n candidates
      - Check if the masked Pokémon appears in top_n
    
    Returns dict of metric values.
    """
    hits = []
    ndcg_scores = []
    precisions = []
    
    for team in test_teams:
        for mask_idx in range(len(team)):
            masked = team[mask_idx]
            partial = [p for i, p in enumerate(team) if i != mask_idx]
            
            recs = model_fn(partial, opponent_team=[])  # no opponent context
            
            hit = masked in recs
            hits.append(int(hit))
            
            # NDCG@n
            relevance = [1 if r == masked else 0 for r in recs]
            ndcg_scores.append(ndcg_at_k(relevance, top_n))
            
            # Precision@3
            precisions.append(int(masked in recs[:3]))
    
    return {
        "hit_rate": sum(hits) / len(hits),
        "ndcg": sum(ndcg_scores) / len(ndcg_scores),
        "precision_at_3": sum(precisions) / len(precisions),
    }


def ndcg_at_k(relevance: list, k: int) -> float:
    """Compute NDCG@k for a single ranked list."""
    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevance[:k])
    )
    ideal = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(sum(relevance), k))
    )
    return dcg / ideal if ideal > 0 else 0.0
```

### 5.3 Beyond-accuracy metrics

```python
def catalog_coverage(all_recommendations: list, legal_pool: list) -> float:
    """What fraction of the legal pool ever gets recommended?"""
    recommended = set(p for recs in all_recommendations for p in recs)
    return len(recommended) / len(legal_pool)


def intra_list_diversity(recs: list, pokemon_types: dict) -> float:
    """Average pairwise type dissimilarity within a recommendation list."""
    if len(recs) < 2:
        return 0.0
    pairs = 0
    dissimilarity_sum = 0.0
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            types_i = set(pokemon_types.get(recs[i], []))
            types_j = set(pokemon_types.get(recs[j], []))
            jaccard = 1 - len(types_i & types_j) / max(len(types_i | types_j), 1)
            dissimilarity_sum += jaccard
            pairs += 1
    return dissimilarity_sum / pairs


def personalization_score(all_recs: list) -> float:
    """
    How different are recommendations across different team inputs?
    1.0 = fully personalised, 0.0 = identical for all inputs.
    """
    n = len(all_recs)
    if n < 2:
        return 0.0
    overlap_sum = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(all_recs[i])
            set_j = set(all_recs[j])
            overlap = len(set_i & set_j) / max(len(set_i | set_j), 1)
            overlap_sum += overlap
            pairs += 1
    return 1 - (overlap_sum / pairs)
```

### 5.4 Simulation-based evaluation (primary business metric)

```python
async def evaluate_model_via_simulation(
    model_fn,
    base_teams: list,           # list of partial teams (5 Pokémon each)
    opponent_teams: list,       # list of opponent teams (0–6 Pokémon each)
    canonical_movesets: dict,
    n_battles: int = 50
) -> pd.DataFrame:
    """
    For each (base_team, opponent_team) pair:
      1. Ask the model to recommend the 6th Pokémon
      2. Build the full team paste
      3. Simulate n_battles
      4. Record win rate
    
    Returns DataFrame comparing win rates across all models.
    """
    results = []
    
    for base_team, opponent in zip(base_teams, opponent_teams):
        rec = model_fn(base_team, opponent)[0]  # top-1 recommendation
        full_team = base_team + [rec]
        paste = build_team_paste(full_team, canonical_movesets)
        win_rate = await evaluate_team(paste, n_battles=n_battles)
        results.append({
            "base_team": base_team,
            "recommended": rec,
            "full_team": full_team,
            "win_rate": win_rate
        })
    
    return pd.DataFrame(results)
```

### 5.5 Full model comparison table

Run all models on the same test scenarios and produce this table:

| Model | NDCG@5 | Precision@3 | Hit Rate@5 | Coverage | Diversity | Personalization | Avg Win Rate |
|---|---|---|---|---|---|---|---|
| Random | | | | | | | |
| Popular | | | | | | | |
| KNN CF (k=10) | | | | | | | |
| KNN CF (k=25) | | | | | | | |
| SVD CF (k=20) | | | | | | | |
| NMF CF (k=20) | | | | | | | |
| Content (stats) | | | | | | | |
| Content (TF-IDF) | | | | | | | |
| Hybrid (no context) | | | | | | | |
| Hybrid + counter | | | | | | | |
| Hybrid + bandit | | | | | | | |

**The win rate column is the headline metric for the CEO slide.**

---

## 6. Notebook Structure

The Jupyter Notebook must be structured in this exact order to tell a coherent story for both the technical evaluator and the CEO presentation.

```
notebook/
  pokecoach.ipynb

data/
  smogon/
    gen9vgc2024regg-1630.txt   (multiple months)
    gen9vgc2024regh-1630.txt   (multiple months)
  pokeapi/
    pokemon.csv
    pokemon_stats.csv
    pokemon_types.csv
    type_efficacy.csv
    pokemon_species.csv
    abilities.csv
    pokemon_abilities.csv
  simulation_results.csv        (generated)
```

### Section 1 — Setup and data loading
- `1.1` Install and import all dependencies
- `1.2` Download Smogon stats files (curl commands + verify)
- `1.3` Download PokeAPI CSVs (curl commands + verify)
- `1.4` Parse Smogon usage tables into DataFrames
- `1.5` Parse Smogon teammate co-occurrence into matrix
- `1.6` Parse Smogon moveset frequency tables (for canonical pastes)
- `1.7` Load PokeAPI tables, join into flat feature table
- `1.8` Build type effectiveness matrix

### Section 2 — Exploratory data analysis
- `2.1` Usage distribution — top 30 Pokémon in Reg G vs Reg H (bar chart)
- `2.2` Meta shift — usage rank changes from Reg G to Reg H (scatter)
- `2.3` Co-occurrence matrix heatmap — top 40 Pokémon (seaborn)
- `2.4` Type distribution of the legal pool (pie/bar)
- `2.5` Sparsity analysis — how many pairs have any co-occurrence data
- `2.6` Speed tier clustering — t-SNE of stat vectors coloured by speed
- `2.7` Top synergy cores — most common Pokémon pairs from co-occurrence

### Section 3 — Preprocessing and feature engineering
- `3.1` Define legal pool for Reg G (filter by usage > 0.5%)
- `3.2` Normalise co-occurrence matrix (row-normalise to [0, 1])
- `3.3` Build content feature matrix (stats + one-hot types + role tags)
- `3.4` Build CVTT split: Reg G = train, Reg H = test
- `3.5` Build evaluation harness: masked team reconstruction task
- `3.6` Build canonical team pastes from Smogon moveset data

### Section 4 — Simulation setup and dataset generation
- `4.1` Verify poke-env and local Showdown server are running
- `4.2` Test one battle with RandomPlayer
- `4.3` Define team configurations to simulate (sample from legal pool)
- `4.4` Run simulation dataset generation (~200 teams × 50 battles)
- `4.5` Analyse simulation results: win rate distribution, team composition patterns
- `4.6` Validate: do high-usage teams win more? (correlation plot)

### Section 5 — Non-personalised baselines
- `5.1` Implement random recommender
- `5.2` Implement popular recommender
- `5.3` Evaluate both on reconstruction task
- `5.4` Evaluate both via simulation (win rate)
- `5.5` Interpret: ceiling and floor established

### Section 6 — Collaborative filtering
- `6.1` Implement memory-based KNN (item-item on co-occurrence)
- `6.2` Vary k (10, 25, 50) — evaluate offline metrics
- `6.3` Implement model-based SVD — vary latent factors (10, 20, 50)
- `6.4` Implement NMF — visualise latent factors as strategy archetypes
- `6.5` t-SNE of Pokémon latent embeddings — colour by archetype
- `6.6` Simulate best CF variant — compare win rate to baselines
- `6.7` Cold-start analysis: how do CF models handle new Reg H Pokémon absent from Reg G training data?

### Section 7 — Content-based recommenders
- `7.1` Implement stat + type content recommender
- `7.2` Implement TF-IDF role recommender (on Smogon analysis text)
- `7.3` Optional: sentence-BERT embeddings for role similarity
- `7.4` NER: extract role tags ("speed control", "trick room setter", "pivot")
- `7.5` Evaluate content models offline and via simulation
- `7.6` Cold-start advantage: content model handles new Pokémon correctly

### Section 8 — Context-aware counter signal
- `8.1` Implement counter scoring function from type_efficacy.csv
- `8.2` Verify: given opponent [Flutter Mane, Iron Bundle] → check top counters make sense
- `8.3` Counter lift metric: NDCG with vs without opponent context
- `8.4` Simulate: does adding opponent context improve win rate?
- `8.5` Visualise: recommendation changes as opponent Pokémon are revealed

### Section 9 — Hybrid recommender
- `9.1` Implement weighted hybrid (α · synergy + β · counter + γ · viability)
- `9.2` Implement context-dependent weight schedule
- `9.3` Implement diversity re-ranking
- `9.4` Grid search over α, β, γ — plot offline metric surface (3D or heatmap)
- `9.5` Full offline evaluation of final hybrid
- `9.6` Simulate hybrid — win rate vs all previous models
- `9.7` Build full model comparison table

### Section 10 — Bandit algorithm
- `10.1` Frame the partial-opponent scenario as an explore-exploit problem
- `10.2` Implement UCB1 over hybrid candidates
- `10.3` Simulate: does UCB1 exploration surface better counters than pure exploitation?
- `10.4` Compare bandit win rate vs deterministic hybrid
- `10.5` Plot: exploration rate vs win rate trade-off

### Section 11 — Results and interpretation
- `11.1` Full metric comparison table (all models × all metrics)
- `11.2` Accuracy vs coverage trade-off plot
- `11.3` Accuracy vs diversity trade-off plot
- `11.4` Meta novelty analysis: does the hybrid recommend beyond just top-usage Pokémon?
- `11.5` CVTT degradation: how much does performance drop on Reg H vs Reg G?
- `11.6` Error analysis: which teams does the system get wrong, and why?
- `11.7` Ethical considerations: bias toward popular archetypes, accessibility for new players

### Section 12 — Interactive prototype
- `12.1` ipywidgets demo: select own team + opponent team → see ranked recommendations in real time
- `12.2` Display score breakdown (synergy / counter / viability contributions) per candidate
- `12.3` Show how recommendations change as opponent Pokémon are revealed

---

## 7. CEO Presentation Outline

Maximum 15 minutes. Non-technical. Lead with the business story.

### Slide 1 — The problem (1 min)
> "Competitive Pokémon players make team-building decisions that can take hours of research. Most players — even experienced ones — rely on guesswork. We built a system that makes those decisions automatically and correctly."

### Slide 2 — What PokéCoach does (2 min)
Live demo of the ipywidgets prototype. Select 5 Pokémon, reveal 2 opponent Pokémon, watch the recommendations update in real time. No jargon. Let the demo do the talking.

### Slide 3 — Why we can trust the recommendations (3 min)
Show the win rate chart. Simple bar chart: Random = 50%, Popular = 56%, Our hybrid = 72% (placeholder — replace with actual numbers). One sentence: "Teams built with our recommendations win X% more battles than teams built without any guidance."

### Slide 4 — How it works (high level only) (3 min)
Three signals, explained in plain English with one example each:
- "It knows which Pokémon are usually on the same team as yours"
- "It knows what beats what the opponent has shown you"
- "It knows which Pokémon are simply performing best in the current meta"

### Slide 5 — What we tried and what worked best (3 min)
One clean table: the model comparison. Highlight that the hybrid beats all individual approaches. Brief note that more data (more regulation sets) would improve confidence.

### Slide 6 — Limitations and next steps (2 min)
- Currently evaluates against an automated opponent, not human players
- Does not model specific EV spreads or niche sets
- Could extend to incorporate real-time opponent scouting

### Slide 7 — Business value (1 min)
> "Millions of players compete in VGC every year. A tool that measurably increases win rates would be used by every serious player. The same recommendation engine could apply to any turn-based strategy game with a large item space."

---

## 8. Dependencies

```txt
# requirements.txt

# Core data science
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# NLP
transformers>=4.30.0    # for optional BERT embeddings
sentence-transformers>=2.2.0

# Simulation
poke-env>=0.8.0
asyncio

# Notebook UI
ipywidgets>=8.0.0
jupyterlab>=4.0.0

# HTTP
requests>=2.28.0

# Optional: dimensionality reduction visualisation
umap-learn>=0.5.0
```

```bash
# Install everything
pip install -r requirements.txt

# Also requires Node.js for the Showdown server
# macOS:  brew install node
# Ubuntu: sudo apt install nodejs npm
# Windows: https://nodejs.org/en/download
```

---

## 9. Key Decisions and Justifications

| Decision | Justification |
|---|---|
| Item-item CF instead of user-item CF | No real user data exists. Item-item CF on Smogon co-occurrence is technically equivalent and produces meaningful "teams that include X also include Y" recommendations |
| Smogon 1630+ cutoff | High-Elo data filters out noise from inexperienced players. Co-occurrence at this level is a genuine win-rate proxy |
| CVTT split by regulation set | Meta shifts between regulation sets are significant enough to constitute distribution shift. Random split would leak future meta knowledge into training |
| SimpleHeuristicsPlayer as opponent | Consistent, deterministic enough for fair comparison. Acknowledged limitation: not equivalent to skilled human play |
| 50 battles per team configuration | Sufficient for ~7% margin of error at 95% confidence. Increasing to 100 halves error but doubles time |
| Context-dependent α/β/γ weights | Hard-coding weights ignores the fact that opponent information fundamentally changes the problem. Dynamic weights are the correct model |
| Diversity re-ranking | Without it, the system can recommend 5 similar Pokémon. Diversity in the recommendation list is independently valuable |

---

## 10. Known Limitations (disclose in methodology)

1. **Opponent quality gap:** Win rates are measured against `SimpleHeuristicsPlayer`, not human players. A team that counters greedy heuristics may perform differently against expert play.

2. **Moveset canonicalisation:** Using the most common moveset per Pokémon ignores situational sets. A Tornadus running Rain Dance instead of Tailwind would perform very differently against certain teams.

3. **Team preview not modelled:** In real VGC, players see all 6 opponent Pokémon during team preview and choose which 4 to bring. The simulation assumes fixed leads.

4. **Stochastic variance:** Pokémon battles involve RNG (accuracy, critical hits, damage rolls). 50 battles per team gives a reasonable estimate but not a perfect win rate.

5. **Meta recency:** Smogon stats lag behind real-time meta shifts. A Pokémon gaining popularity mid-season won't be reflected until the next monthly stats file.