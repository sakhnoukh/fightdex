from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "").replace("'", "")


def _filter_candidates(candidates: list[str], banned: set[str], k: int) -> list[str]:
    return [c for c in candidates if c not in banned][:k]


@dataclass
class ModelData:
    legal_pool: pd.DataFrame
    cooccurrence: pd.DataFrame
    content: pd.DataFrame
    counter: pd.DataFrame
    moveset: pd.DataFrame
    name_map: dict[str, str]
    pokemon_types: dict[str, list[str]] = field(default_factory=dict)


def load_model_data(artifacts_root: Path) -> ModelData:
    legal = pd.read_csv(artifacts_root / "features" / "legal_pool.csv")
    cooc = pd.read_csv(artifacts_root / "features" / "cooccurrence.csv").set_index("pokemon")
    content = pd.read_csv(artifacts_root / "features" / "content_features.csv")
    counter = pd.read_csv(artifacts_root / "features" / "counter_matrix.csv").set_index("pokemon")
    moveset = pd.read_csv(artifacts_root / "smogon" / "regg_moveset.csv")
    map_path = artifacts_root / "mappings" / "pokemon_name_map.json"
    name_map = {}
    if map_path.exists():
        name_map = pd.read_json(map_path, typ="series").to_dict()
    types_path = artifacts_root / "features" / "pokemon_types.json"
    pokemon_types: dict[str, list[str]] = {}
    if types_path.exists():
        pokemon_types = json.loads(types_path.read_text())
    return ModelData(legal, cooc, content, counter, moveset, name_map, pokemon_types)


def recommend_moveset(pokemon: str, moveset_df: pd.DataFrame, k: int = 4) -> list[str]:
    """Return top-k moves for a pokemon ranked by usage_pct."""
    subset = moveset_df[moveset_df["pokemon"] == pokemon]
    if subset.empty:
        return []
    return subset.nlargest(k, "usage_pct")["move"].tolist()


class BaseModel:
    def __init__(self, data: ModelData) -> None:
        self.data = data
        self.items = data.legal_pool["pokemon"].tolist()
        self.random = random.Random(42)

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        raise NotImplementedError


class RandomRecommender(BaseModel):
    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        candidates = [x for x in self.items if x not in set(partial_team)]
        self.random.shuffle(candidates)
        return candidates[:k]


class PopularityRecommender(BaseModel):
    def __init__(self, data: ModelData) -> None:
        super().__init__(data)
        self.rank = (
            data.legal_pool.sort_values("usage_pct", ascending=False)["pokemon"].tolist()
        )

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        return _filter_candidates(self.rank, set(partial_team), k)


class KNNRecommender(BaseModel):
    def __init__(self, data: ModelData) -> None:
        super().__init__(data)
        self.cooc = data.cooccurrence.reindex(index=self.items, columns=self.items, fill_value=0.0)

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        scores = pd.Series(0.0, index=self.items)
        for mon in partial_team:
            if mon in self.cooc.index:
                scores += self.cooc.loc[mon]
        ranked = scores.sort_values(ascending=False).index.tolist()
        return _filter_candidates(ranked, set(partial_team), k)


class MatrixFactorizationRecommender(BaseModel):
    def __init__(self, data: ModelData, method: str = "svd", components: int = 24) -> None:
        super().__init__(data)
        matrix = data.cooccurrence.reindex(index=self.items, columns=self.items, fill_value=0.0).values
        if method == "nmf":
            model = NMF(n_components=min(components, matrix.shape[0] - 1), init="nndsvda", random_state=42, max_iter=500)
            transformed = model.fit_transform(np.clip(matrix, a_min=0.0, a_max=None))
            reconstructed = np.dot(transformed, model.components_)
        else:
            model = TruncatedSVD(n_components=min(components, matrix.shape[0] - 1), random_state=42)
            transformed = model.fit_transform(matrix)
            reconstructed = np.dot(transformed, model.components_)
        self.reconstructed = pd.DataFrame(reconstructed, index=self.items, columns=self.items)

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        scores = pd.Series(0.0, index=self.items)
        for mon in partial_team:
            if mon in self.reconstructed.index:
                scores += self.reconstructed.loc[mon]
        ranked = scores.sort_values(ascending=False).index.tolist()
        return _filter_candidates(ranked, set(partial_team), k)


class ContentRecommender(BaseModel):
    def __init__(self, data: ModelData) -> None:
        super().__init__(data)
        content = data.content.copy()
        content["norm_name"] = content["pokemon"].map(lambda x: x.replace("_", "-"))
        stats_cols = [c for c in content.columns if c not in {"pokemon", "norm_name"}]
        matrix = content[stats_cols].values
        sims = cosine_similarity(matrix)
        self.sim = pd.DataFrame(sims, index=content["norm_name"], columns=content["norm_name"])

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        scores = pd.Series(0.0, index=self.sim.columns)
        for mon in partial_team:
            norm = _normalize_name(mon)
            if norm in self.sim.index:
                scores += self.sim.loc[norm]
        ranked_norm = scores.sort_values(ascending=False).index.tolist()
        inv = {v: k for k, v in self.data.name_map.items()} if self.data.name_map else {}
        ranked = [inv.get(n, n) for n in ranked_norm]
        return _filter_candidates(ranked, set(partial_team), k)


class TfidfRoleRecommender(BaseModel):
    def __init__(self, data: ModelData) -> None:
        super().__init__(data)
        docs: dict[str, str] = {}
        for mon, group in data.moveset.groupby("pokemon"):
            top_moves = group.sort_values("usage_pct", ascending=False)["move"].head(10).tolist()
            docs[mon] = " ".join(top_moves)
        self.mon_list = list(docs.keys())
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf = vectorizer.fit_transform([docs[m] for m in self.mon_list])
        self.sim = cosine_similarity(self.tfidf)
        self.idx = {m: i for i, m in enumerate(self.mon_list)}

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        scores = np.zeros(len(self.mon_list))
        for mon in partial_team:
            if mon in self.idx:
                scores += self.sim[self.idx[mon]]
        ranked_idx = np.argsort(scores)[::-1]
        ranked = [self.mon_list[i] for i in ranked_idx]
        return _filter_candidates(ranked, set(partial_team), k)


class HybridRecommender(BaseModel):
    def __init__(
        self,
        data: ModelData,
        alpha: float = 0.45,
        beta: float = 0.30,
        gamma: float = 0.25,
    ) -> None:
        super().__init__(data)
        self.knn = KNNRecommender(data)
        self.content = ContentRecommender(data)
        self.pop = PopularityRecommender(data)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.counter = data.counter.reindex(index=self.items, columns=self.items, fill_value=1.0)

    def _context_weights(self, partial_team: list[str], opponent_context: list[str] | None) -> tuple[float, float, float]:
        if not partial_team:
            return 0.15, 0.0, 0.85
        if opponent_context:
            return 0.45, 0.35, 0.20
        return self.alpha, self.beta, self.gamma

    def recommend(
        self,
        partial_team: list[str],
        opponent_context: list[str] | None = None,
        k: int = 5,
        type_filter: list[str] | None = None,
        weight_overrides: dict | None = None,
    ) -> list[str]:
        if weight_overrides:
            alpha = float(weight_overrides.get("synergy", self.alpha))
            beta = float(weight_overrides.get("counter", self.beta))
            gamma = float(weight_overrides.get("viability", self.gamma))
            content_w = float(weight_overrides.get("content", 0.15))
        else:
            alpha, beta, gamma = self._context_weights(partial_team, opponent_context)
            content_w = 0.15

        synergy_rank = self.knn.recommend(partial_team, k=len(self.items))
        viability_rank = self.pop.recommend(partial_team, k=len(self.items))
        content_rank = self.content.recommend(partial_team, k=len(self.items))

        def to_score(rank: list[str]) -> pd.Series:
            scores = pd.Series(0.0, index=self.items)
            for i, mon in enumerate(rank):
                if mon in scores.index:
                    scores.loc[mon] = 1.0 / (i + 1)
            return scores

        synergy_scores = to_score(synergy_rank)
        viability_scores = to_score(viability_rank)
        content_scores = to_score(content_rank)
        counter_scores = pd.Series(0.0, index=self.items)
        if opponent_context:
            for opp in opponent_context:
                if opp in self.counter.columns:
                    counter_scores += self.counter[opp]
            counter_scores /= max(1, len(opponent_context))

        scores = alpha * synergy_scores + gamma * viability_scores + content_w * content_scores
        scores += beta * counter_scores
        ranked = scores.sort_values(ascending=False).index.tolist()

        # Apply type filter: keep only pokemon of the requested types (with fallback)
        if type_filter and self.data.pokemon_types:
            type_set = {t.lower() for t in type_filter}
            filtered = [p for p in ranked if type_set & set(self.data.pokemon_types.get(p, []))]
            if filtered:
                ranked = filtered

        base = _filter_candidates(ranked, set(partial_team), k * 3)
        diverse: list[str] = []
        for cand in base:
            if not diverse:
                diverse.append(cand)
                if len(diverse) >= k:
                    break
                continue
            too_similar = False
            for chosen in diverse:
                if chosen in self.counter.index and cand in self.counter.columns:
                    if float(self.counter.loc[chosen, cand]) > 2.0:
                        too_similar = True
                        break
            if not too_similar:
                diverse.append(cand)
            if len(diverse) >= k:
                break
        return diverse[:k]


class UCBHybridRecommender(BaseModel):
    def __init__(self, data: ModelData, c: float = 1.2) -> None:
        super().__init__(data)
        self.hybrid = HybridRecommender(data)
        self.counts = {m: 1 for m in self.items}
        self.rewards = {m: 0.5 for m in self.items}
        self.c = c
        self.t = 1

    def update_reward(self, mon: str, reward: float) -> None:
        self.counts[mon] = self.counts.get(mon, 1) + 1
        self.rewards[mon] = self.rewards.get(mon, 0.0) + reward
        self.t += 1

    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        base = self.hybrid.recommend(partial_team, opponent_context=opponent_context, k=max(12, k))
        scored: list[tuple[str, float]] = []
        for mon in base:
            mean = self.rewards.get(mon, 0.5) / self.counts.get(mon, 1)
            bonus = self.c * math.sqrt(math.log(self.t + 1) / self.counts.get(mon, 1))
            scored.append((mon, mean + bonus))
        scored.sort(key=lambda x: x[1], reverse=True)
        return _filter_candidates([m for m, _ in scored], set(partial_team), k)


def build_model_suite(data: ModelData) -> dict[str, BaseModel]:
    return {
        "random": RandomRecommender(data),
        "popular": PopularityRecommender(data),
        "knn_cf": KNNRecommender(data),
        "svd_mf": MatrixFactorizationRecommender(data, method="svd"),
        "nmf_mf": MatrixFactorizationRecommender(data, method="nmf"),
        "content": ContentRecommender(data),
        "tfidf": TfidfRoleRecommender(data),
        "hybrid": HybridRecommender(data),
        "bandit_hybrid": UCBHybridRecommender(data),
    }
