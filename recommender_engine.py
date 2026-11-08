"""
Hybride K-Means + Content-Based Recommender Engine
Gebruikt de gewranglede & geclusterde dataset om razendsnel gepersonaliseerde
filmaanbevelingen te berekenen inclusief uitleg ('Waarom deze film?').
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel


BASE_DIR = Path(__file__).resolve().parent
CLUSTERED_CSV = BASE_DIR / "movies_wrangled_clustered.csv"
RAW_CSV = BASE_DIR / "movies_dataset.csv"


CLUSTER_META = {
    "Blockbusters & Mega-Hits": {
        "color": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.16)",
        "description": "Grootschalige bioscoophits met hoge budgetten, veel publieksbereik en sterke opbrengsten.",
    },
    "Mid-Budget Mainstream": {
        "color": "#3b82f6",
        "badge_bg": "rgba(59, 130, 246, 0.16)",
        "description": "Toegankelijke commerciële speelfilms (vaak comedy, actie of avontuur) met solide opbrengsten.",
    },
    "Hooggewaardeerde Drama & Prestige": {
        "color": "#a855f7",
        "badge_bg": "rgba(168, 85, 247, 0.16)",
        "description": "Kritisch geprezen films en klassiekers met langere speelduur en hoge publieksscores.",
    },
    "Low-Budget / Indie Hits": {
        "color": "#10b981",
        "badge_bg": "rgba(16, 185, 129, 0.16)",
        "description": "Creatieve films met een bescheiden budget die relatief veel opbrengen en goed worden gewaardeerd.",
    },
    "Box-Office Flops": {
        "color": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.16)",
        "description": "Films die in de bioscoop hun budget niet terugverdienden — inclusief cultfilms en streaming-releases.",
    },
}


def ensure_enriched_dataset() -> pd.DataFrame:
    """
    Zorgt ervoor dat movies_wrangled_clustered.csv ook tekstuele metadata en poster-paden bevat
    voor het recommender system. Verrijkt eenmalig vanuit movies_dataset.csv indien nodig.
    """
    if not CLUSTERED_CSV.exists():
        raise FileNotFoundError(
            f"{CLUSTERED_CSV.name} niet gevonden. Draai eerst movies_kmeans.py."
        )

    df = pd.read_csv(CLUSTERED_CSV, encoding="utf-8")
    needed_cols = {"overview", "keywords", "top_cast", "poster_path", "tagline"}
    if needed_cols.issubset(df.columns):
        return df

    print("Eenmalig verrijken van geclusterde dataset met plot, cast, keywords en posters...")
    extra_cols = [
        "id",
        "overview",
        "tagline",
        "keywords",
        "credits",
        "production_companies",
        "poster_path",
        "backdrop_path",
    ]
    raw_extra = pd.read_csv(
        RAW_CSV,
        usecols=extra_cols,
        encoding="utf-8",
        low_memory=False,
    ).drop_duplicates(subset=["id"], keep="first")

    raw_extra["top_cast"] = (
        raw_extra["credits"]
        .fillna("")
        .astype(str)
        .apply(lambda s: "-".join([p.strip() for p in s.split("-") if p.strip()][:5]))
    )
    raw_extra["top_studios"] = (
        raw_extra["production_companies"]
        .fillna("")
        .astype(str)
        .apply(lambda s: "-".join([p.strip() for p in s.split("-") if p.strip()][:2]))
    )
    raw_extra = raw_extra.drop(columns=["credits", "production_companies"])

    # Verwijder eventueel al deels aanwezige kolommen voor een schone merge
    overlap = [c for c in raw_extra.columns if c in df.columns and c != "id"]
    if overlap:
        df = df.drop(columns=overlap)

    df = df.merge(raw_extra, on="id", how="left")
    for col in ["overview", "tagline", "keywords", "top_cast", "top_studios", "poster_path", "backdrop_path"]:
        df[col] = df[col].fillna("")

    df.to_csv(CLUSTERED_CSV, index=False, encoding="utf-8")
    print(f"Verrijkte dataset opgeslagen ({len(df):,} films).")
    return df


class MovieRecommenderEngine:
    """Hybride Recommender op basis van K-Means clusters + TF-IDF Cosine Similarity."""

    KMEANS_FEATURES = [
        "log_budget",
        "log_revenue",
        "log_roi",
        "runtime",
        "vote_average",
        "log_vote_count",
        "log_popularity",
    ]

    def __init__(self) -> None:
        self.df = ensure_enriched_dataset().reset_index(drop=True)
        for col in ["overview", "tagline", "keywords", "top_cast", "top_studios", "poster_path", "backdrop_path", "genres"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        self.id_to_idx = {int(mid): idx for idx, mid in enumerate(self.df["id"])}
        self._build_matrices()
        self._prepare_map_sample()

    def _build_matrices(self) -> None:
        # 1. Tekstuele content-soup samenstellen (gewogen herhaling voor genres, keywords & cast)
        genres_clean = self.df["genres"].str.replace("-", " ", regex=False)
        keywords_clean = self.df["keywords"].str.replace("-", " ", regex=False)
        cast_token = self.df["top_cast"].apply(
            lambda s: " ".join(p.strip().lower().replace(" ", "_") for p in s.split("-") if p.strip())
        )
        overview_clean = self.df["overview"]

        soup = (
            (genres_clean + " ") * 3
            + (keywords_clean + " ") * 2
            + (cast_token + " ") * 2
            + overview_clean
        )

        self.tfidf = TfidfVectorizer(
            max_features=7500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(soup)

        # 2. Numerieke K-Means feature-matrix: alléén kolomgewijze StandardScaler (Z-scores),
        #    zonder rij-normalisatie naar lengte 1, zodat Euclidische (L2) afstand behouden blijft.
        scaler = StandardScaler()
        self.kmeans_matrix = scaler.fit_transform(self.df[self.KMEANS_FEATURES])

        # 3. Kwaliteitsscore (0..1) als lichte tie-breaker
        raw_q = self.df["vote_average"] * self.df["log_vote_count"]
        self.quality_norm = ((raw_q - raw_q.min()) / (raw_q.max() - raw_q.min())).to_numpy()

        # Pre-parse sets voor snelle 'Waarom aanbevolen?'-uitleg
        self.genre_sets = [
            set(g.strip() for g in s.split("-") if g.strip()) for s in self.df["genres"]
        ]
        self.keyword_sets = [
            set(k.strip() for k in s.split("-") if k.strip()) for s in self.df["keywords"]
        ]
        self.cast_sets = [
            set(c.strip() for c in s.split("-") if c.strip()) for s in self.df["top_cast"]
        ]

    def _prepare_map_sample(self) -> None:
        """Bereid een gestratificeerde steekproef voor voor de interactieve 2D K-Means PCA-kaart."""
        parts = []
        for _, grp in self.df.groupby("cluster_naam"):
            top_part = grp.nlargest(25, "vote_count")
            rest = grp.drop(top_part.index)
            rand_part = rest.sample(min(75, len(rest)), random_state=42)
            parts.append(pd.concat([top_part, rand_part]))
        sample_df = pd.concat(parts, ignore_index=True)

        self.pca_background = [
            {
                "id": int(r.id),
                "title": r.title,
                "year": int(r.release_year),
                "cluster": r.cluster_naam,
                "pc1": round(float(r.pc1), 3),
                "pc2": round(float(r.pc2), 3),
                "rating": round(float(r.vote_average), 1),
            }
            for r in sample_df.itertuples()
        ]

    def get_metadata(self) -> dict:
        genres = sorted(
            {
                g.strip()
                for s in self.df["genres"]
                for g in s.split("-")
                if g.strip() and g.strip() != "TV Movie"
            }
        )
        clusters = []
        for cname, meta in CLUSTER_META.items():
            sub = self.df[self.df["cluster_naam"] == cname]
            clusters.append(
                {
                    "name": cname,
                    "count": int(len(sub)),
                    "color": meta["color"],
                    "badge_bg": meta["badge_bg"],
                    "description": meta["description"],
                    "med_budget_m": round(float(sub["budget"].median() / 1e6), 2),
                    "med_revenue_m": round(float(sub["revenue"].median() / 1e6), 2),
                    "med_roi": round(float(sub["roi_ratio"].median()), 2),
                    "avg_rating": round(float(sub["vote_average"].mean()), 2),
                }
            )

        featured_titles = [
            "Inception",
            "The Dark Knight",
            "Pulp Fiction",
            "Interstellar",
            "The Blair Witch Project",
            "Amadeus",
            "Home Alone",
            "Mad Max: Fury Road",
        ]
        featured = []
        for t in featured_titles:
            m = self.df[self.df["title"].str.lower() == t.lower()].sort_values(
                "vote_count", ascending=False
            )
            if not m.empty:
                featured.append(self.serialize_movie(int(m.index[0])))

        return {
            "total_movies": int(len(self.df)),
            "genres": genres,
            "clusters": clusters,
            "featured": featured,
            "year_min": int(self.df["release_year"].min()),
            "year_max": int(self.df["release_year"].max()),
            "pca_background": self.pca_background,
        }

    def serialize_movie(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        cname = str(row["cluster_naam"])
        cmeta = CLUSTER_META.get(cname, {"color": "#64748b", "badge_bg": "rgba(100,116,139,0.16)"})
        return {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "year": int(row["release_year"]),
            "genres": [g for g in str(row["genres"]).split("-") if g],
            "primary_genre": str(row["primary_genre"]),
            "overview": str(row["overview"]),
            "tagline": str(row["tagline"]),
            "cast": [c for c in str(row["top_cast"]).split("-") if c][:4],
            "studios": [s for s in str(row["top_studios"]).split("-") if s][:2],
            "budget_m": round(float(row["budget"]) / 1e6, 2),
            "revenue_m": round(float(row["revenue"]) / 1e6, 2),
            "roi": round(float(row["roi_ratio"]), 2),
            "runtime": int(round(float(row["runtime"]))),
            "rating": round(float(row["vote_average"]), 1),
            "votes": int(round(float(row["vote_count"]))),
            "cluster": cname,
            "cluster_color": cmeta["color"],
            "cluster_bg": cmeta["badge_bg"],
            "poster_url": f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if row["poster_path"] else "",
            "backdrop_url": f"https://image.tmdb.org/t/p/w780{row['backdrop_path']}" if row["backdrop_path"] else "",
            "pc1": round(float(row["pc1"]), 3),
            "pc2": round(float(row["pc2"]), 3),
        }

    def search(self, query: str, limit: int = 12) -> list[dict]:
        q = query.strip().lower()
        if not q:
            top_idx = self.df.nlargest(limit, "vote_count").index
            return [self.serialize_movie(int(i)) for i in top_idx]

        titles_lower = self.df["title"].str.lower()
        starts = titles_lower.str.startswith(q, na=False)
        contains = titles_lower.str.contains(q, regex=False, na=False)
        cast_match = self.df["top_cast"].str.lower().str.contains(q, regex=False, na=False)

        candidates = self.df[starts | contains | cast_match].copy()
        if candidates.empty:
            return []

        candidates["_rank"] = (
            starts.loc[candidates.index].astype(int) * 3
            + contains.loc[candidates.index].astype(int) * 2
            + np.log10(candidates["vote_count"] + 1) * 0.4
        )
        best = candidates.sort_values("_rank", ascending=False).head(limit)
        return [self.serialize_movie(int(i)) for i in best.index]

    def recommend(
        self,
        seed_ids: list[int],
        content_weight: float = 0.65,
        cluster_mode: str = "hybrid",
        genre: str = "",
        min_rating: float = 0.0,
        year_min: int = 1920,
        year_max: int = 2026,
        max_runtime: int = 300,
        limit: int = 12,
    ) -> dict:
        valid_indices = [self.id_to_idx[mid] for mid in seed_ids if mid in self.id_to_idx]
        if not valid_indices:
            fallback = self.df.nlargest(limit, "vote_count").index
            return {
                "seeds": [],
                "taste_profile": None,
                "recommendations": [self.serialize_movie(int(i)) for i in fallback],
            }

        content_weight = float(np.clip(content_weight, 0.0, 1.0))
        kmeans_weight = 1.0 - content_weight

        # 1. Gemiddelde TF-IDF cosine similarity tot alle gekozen seed-films
        seed_tfidf = self.tfidf_matrix[valid_indices]
        content_sims = np.asarray(linear_kernel(seed_tfidf, self.tfidf_matrix).mean(axis=0)).ravel()
        if content_sims.max() > 0:
            content_sims_norm = content_sims / content_sims.max()
        else:
            content_sims_norm = content_sims

        # 2. K-Means 7D numerieke nabijheid op basis van standaard Euclidische (L2) afstand
        seed_km = self.kmeans_matrix[valid_indices]
        km_l2_dists = euclidean_distances(seed_km, self.kmeans_matrix).mean(axis=0)
        km_sims_norm = 1.0 / (1.0 + km_l2_dists)

        # 3. Cluster-affiniteit bonus
        seed_clusters = set(self.df.loc[valid_indices, "cluster_naam"])
        in_seed_cluster = self.df["cluster_naam"].isin(seed_clusters).to_numpy(dtype=float)

        combined_score = (
            content_weight * content_sims_norm
            + kmeans_weight * km_sims_norm
            + (0.06 * in_seed_cluster if cluster_mode == "hybrid" else 0.0)
            + 0.04 * self.quality_norm
        )

        # 4. Filters toepassen
        mask = np.ones(len(self.df), dtype=bool)
        mask[valid_indices] = False

        if cluster_mode == "same":
            mask &= self.df["cluster_naam"].isin(seed_clusters).to_numpy()
        elif cluster_mode not in ("hybrid", "all", ""):
            mask &= (self.df["cluster_naam"] == cluster_mode).to_numpy()

        if genre:
            mask &= self.df["genres"].str.contains(genre, regex=False, na=False).to_numpy()
        if min_rating > 0:
            mask &= (self.df["vote_average"] >= min_rating).to_numpy()
        mask &= self.df["release_year"].between(year_min, year_max).to_numpy()
        mask &= (self.df["runtime"] <= max_runtime).to_numpy()

        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0:
            return {
                "seeds": [self.serialize_movie(i) for i in valid_indices],
                "taste_profile": self._build_taste_profile(valid_indices),
                "recommendations": [],
            }

        top_order = candidate_indices[np.argsort(combined_score[candidate_indices])[::-1][:limit]]

        # Verzameling van seed-kenmerken voor uitlegbare aanbevelingen
        union_genres = set().union(*(self.genre_sets[i] for i in valid_indices))
        union_keywords = set().union(*(self.keyword_sets[i] for i in valid_indices))
        union_cast = set().union(*(self.cast_sets[i] for i in valid_indices))

        recs = []
        max_raw = max(float(combined_score[top_order[0]]), 1e-6)
        for idx in top_order:
            item = self.serialize_movie(int(idx))
            raw_s = float(combined_score[idx])
            # Schaal naar intuïtief match-percentage (55% - 98%)
            match_pct = int(np.clip(round(52 + 46 * (raw_s / max(max_raw, 0.45))), 50, 99))

            shared_genres = sorted(union_genres & self.genre_sets[idx])[:3]
            shared_kw = sorted(union_keywords & self.keyword_sets[idx])[:3]
            shared_actors = sorted(union_cast & self.cast_sets[idx])[:2]

            reasons = []
            if item["cluster"] in seed_clusters:
                reasons.append(f"Zelfde categorie: {item['cluster']}")
            else:
                reasons.append(f"Andere categorie: {item['cluster']}")
            if shared_actors:
                reasons.append(f"Met: {', '.join(shared_actors)}")
            if shared_kw:
                reasons.append(f"Onderwerp: {', '.join(shared_kw)}")
            elif shared_genres:
                reasons.append(f"Genre: {', '.join(shared_genres)}")

            item["match_pct"] = match_pct
            item["content_score"] = round(float(content_sims_norm[idx]) * 100, 1)
            item["kmeans_score"] = round(float(km_sims_norm[idx]) * 100, 1)
            item["reasons"] = reasons
            recs.append(item)

        return {
            "seeds": [self.serialize_movie(i) for i in valid_indices],
            "taste_profile": self._build_taste_profile(valid_indices),
            "recommendations": recs,
        }

    def _build_taste_profile(self, valid_indices: list[int]) -> dict:
        sub = self.df.iloc[valid_indices]
        dominant_cluster = str(sub["cluster_naam"].mode().iloc[0])
        cmeta = CLUSTER_META.get(dominant_cluster, {"color": "#64748b"})
        all_genres = [g for i in valid_indices for g in self.genre_sets[i]]
        top_genres = pd.Series(all_genres).value_counts().head(3).index.tolist() if all_genres else []
        return {
            "dominant_cluster": dominant_cluster,
            "cluster_color": cmeta["color"],
            "avg_budget_m": round(float(sub["budget"].mean() / 1e6), 1),
            "avg_rating": round(float(sub["vote_average"].mean()), 2),
            "avg_runtime": int(round(float(sub["runtime"].mean()))),
            "top_genres": top_genres,
            "centroid_pc1": round(float(sub["pc1"].mean()), 3),
            "centroid_pc2": round(float(sub["pc2"].mean()), 3),
        }

