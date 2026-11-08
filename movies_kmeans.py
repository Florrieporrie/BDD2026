"""
Data Wrangling, K-Means Clustering & Visualisatie voor movies_dataset.csv
Vak: Big Data Design - Lab opdracht

Methodologische kernkeuzes voor K-Means:
- Kolomgewijze standaardisatie via StandardScaler (Z-score per variabele: gem. = 0, std = 1),
  zodat schaalverschillen tussen variabelen (bijv. minuten vs. log-dollars vs. ratings) de
  afstandsberekening niet vertekenen, terwijl de individuele vectorlengtes behouden blijven.
- Traditionele K-Means (sklearn.cluster.KMeans met k=5) op basis van de standaard
  Euclidische (L2) afstand tot cluster-centroïden en minimalisatie van Inertia (WCSS).
- Géén rij-normalisatie naar vectorlengte 1 en géén cosine similarity: zowel richting als
  absolute omvang (magnitude) in de gestandaardiseerde kenmerkruimte tellen mee.
"""

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Automatische controle en installatie van benodigde packages indien afwezig
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
}

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"Pakket '{pip_name}' ontbreekt. Bezig met automatische installatie...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "movies_dataset.csv"
OUTPUT_CSV = BASE_DIR / "movies_wrangled_clustered.csv"
OUTPUT_PLOT = BASE_DIR / "kmeans_visualisatie.png"


def load_and_wrangle(csv_path: Path) -> pd.DataFrame:
    print(f"[1/4] Inladen van {csv_path.name}...")
    usecols = [
        "id",
        "title",
        "genres",
        "original_language",
        "popularity",
        "release_date",
        "budget",
        "revenue",
        "runtime",
        "status",
        "vote_average",
        "vote_count",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    n_raw = len(df)
    print(f"      Ruwe dataset: {n_raw:,} rijen")

    # 1. Deduplicatie op film-ID
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    n_dedup = len(df)
    print(f"      Na verwijderen duplicaten op 'id': {n_dedup:,} rijen (-{n_raw - n_dedup:,})")

    # 2. Alleen daadwerkelijk uitgebrachte films met een geldige titel
    df = df[df["status"] == "Released"].dropna(subset=["title"]).copy()

    # 3. Datatypes opschonen & datum parsen
    num_cols = ["popularity", "budget", "revenue", "runtime", "vote_average", "vote_count"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # 4. Filteren van impliciete missings (0-waarden in TMDB) en onrealistische waarden
    #    - Budget & revenue >= $10.000 (verwijdert 0-placeholders en $1 test-entries)
    #    - Speelduur tussen 40 en 300 minuten (echte speelfilms)
    #    - Minimaal 25 stemmen zodat vote_average statistisch betrouwbaar is
    mask = (
        (df["budget"] >= 10_000)
        & (df["revenue"] >= 10_000)
        & (df["runtime"].between(40, 300))
        & (df["vote_count"] >= 25)
        & (df["vote_average"].between(1.0, 10.0))
        & (df["popularity"] > 0)
        & (df["release_year"].between(1920, 2026))
    )
    df = df.loc[mask].copy()
    print(f"      Na kwaliteits- en validiteitsfilters: {len(df):,} speelfilms over")

    # 5. Feature engineering op financiële en beoordelingsstatistieken
    df["primary_genre"] = (
        df["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.split("-")
        .str[0]
        .replace("", "Unknown")
    )
    df["roi_ratio"] = df["revenue"] / df["budget"]
    df["profit_musd"] = (df["revenue"] - df["budget"]) / 1e6

    # Cap extreme ROI uitschieters op het 99.5e percentiel voor stabiele Euclidische centroïden
    roi_cap = df["roi_ratio"].quantile(0.995)
    df["roi_capped"] = df["roi_ratio"].clip(upper=roi_cap)

    # Log10-transformaties om sterke rechts-scheefheid (skewness) van geldbedragen en aantallen
    # te verminderen voordat de Euclidische afstand wordt berekend
    df["log_budget"] = np.log10(df["budget"])
    df["log_revenue"] = np.log10(df["revenue"])
    df["log_roi"] = np.log10(df["roi_capped"])
    df["log_vote_count"] = np.log10(df["vote_count"])
    df["log_popularity"] = np.log10(df["popularity"] + 1.0)

    return df.reset_index(drop=True)


def assign_cluster_names(df: pd.DataFrame) -> dict:
    """Ken consistente, inhoudelijke namen toe op basis van de cluster-statistieken."""
    summary = df.groupby("cluster_id").agg(
        med_budget=("budget", "median"),
        med_roi=("roi_ratio", "median"),
        mean_rating=("vote_average", "mean"),
    )
    remaining = set(summary.index)

    blockbuster_cid = summary.loc[list(remaining), "med_budget"].idxmax()
    remaining.remove(blockbuster_cid)

    flop_cid = summary.loc[list(remaining), "med_roi"].idxmin()
    remaining.remove(flop_cid)

    indie_cid = summary.loc[list(remaining), "med_budget"].idxmin()
    remaining.remove(indie_cid)

    arthouse_cid = summary.loc[list(remaining), "mean_rating"].idxmax()
    remaining.remove(arthouse_cid)

    mainstream_cid = remaining.pop()

    return {
        blockbuster_cid: "Blockbusters & Mega-Hits",
        mainstream_cid: "Mid-Budget Mainstream",
        arthouse_cid: "Hooggewaardeerde Drama & Prestige",
        indie_cid: "Low-Budget / Indie Hits",
        flop_cid: "Box-Office Flops",
    }


def run_kmeans(df: pd.DataFrame, k_final: int = 5):
    print("[2/4] Standaardiseren van features & evalueren van K-Means (k=2..9)...")
    features = [
        "log_budget",
        "log_revenue",
        "log_roi",
        "runtime",
        "vote_average",
        "log_vote_count",
        "log_popularity",
    ]

    # -------------------------------------------------------------------------
    # WAAROM STANDARDSCALER (EN GEEN RIJ-NORMALISATIE TOT LENGTE 1)?
    # K-Means berekent de Euclidische (L2) afstand tussen datapunten en centroïden.
    # Zonder schaling zou een variabele met een groot getalsbereik (zoals 'runtime'
    # in tientallen/honderden minuten) veel zwaarder meetellen in de afstand dan
    # 'vote_average' of 'log_budget'.
    #
    # StandardScaler past kolomgewijze Z-score standaardisatie toe (gemiddelde = 0,
    # standaardafwijking = 1 per feature). Belangrijk: dit normaliseert alléén de
    # variantie per kolom en forceert NIET de lengte (L2-norm) van elke rijvector
    # naar 1 (zoals sklearn.preprocessing.Normalizer zou doen). Zou je rijvectoren
    # wél tot lengte 1 schalen, dan geldt ||u - v||_2^2 = 2 * (1 - cos(u, v)) en
    # boots je feitelijk cosine similarity na, waardoor het verschil in schaal/omvang
    # tussen bijv. een mega-blockbuster en een gemiddelde film verloren gaat.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled_df = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index,
    )

    ks = list(range(2, 10))
    inertias = []
    silhouettes = []

    for k in ks:
        # sklearn.cluster.KMeans hanteert inherent de standaard Euclidische (L2) afstand:
        # elke film wordt toegewezen aan het dichtstbijzijnde centroïde in Euclidische zin,
        # en minimaliseert de Within-Cluster Sum of Squares (Inertia = som van gekwadrateerde
        # Euclidische afstanden tot het cluster-centroïde).
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=15)
        labels = km.fit_predict(X_scaled_df)
        inertias.append(km.inertia_)

        # Ook bij de Silhouette-validatie gebruiken we expliciet metric="euclidean" (L2),
        # volledig consistent met de Euclidische afstandsmaat van K-Means (géén cosine).
        sil = silhouette_score(
            X_scaled_df,
            labels,
            metric="euclidean",
            sample_size=min(5000, len(df)),
            random_state=42,
        )
        silhouettes.append(sil)
        print(f"      k={k}: Inertia (WCSS)={km.inertia_:,.1f} | Silhouette (Euclidisch)={sil:.3f}")

    # -------------------------------------------------------------------------
    # DEFINITIEVE K-MEANS CLUSTERING (k=5) OP BASIS VAN EUCLIDISCHE (L2) AFSTAND
    # k=5 combineert een duidelijke knik in de Inertia-curve (WCSS) met een lokale
    # piek in de Euclidische Silhouette-score. Het algoritme gebruikt uitsluitend de
    # standaard Euclidische afstand tot de centroïden (kmeans.cluster_centers_),
    # zonder enige vorm van cosine similarity of sferische normalisatie.
    # -------------------------------------------------------------------------
    print(f"[3/4] Definitieve K-Means fitten met k={k_final} (standaard Euclidische L2-afstand)...")
    kmeans = KMeans(n_clusters=k_final, init="k-means++", random_state=42, n_init=25)
    df["cluster_id"] = kmeans.fit_predict(X_scaled_df)

    # Bereken via kmeans.transform() de exacte Euclidische (L2) afstand van elke film
    # tot zijn eigen toegewezen cluster-centroïde ter controle en transparantie.
    all_centroid_distances = kmeans.transform(X_scaled_df)
    df["centroid_l2_distance"] = all_centroid_distances[
        np.arange(len(df)), df["cluster_id"].to_numpy()
    ]
    print(f"      Definitieve Inertia (Within-Cluster Sum of Squares): {kmeans.inertia_:,.2f}")

    # 2D PCA voor visualisatie van de 7D gestandaardiseerde Euclidische ruimte
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled_df)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    centroids_pca = pca.transform(
        pd.DataFrame(kmeans.cluster_centers_, columns=features)
    )

    label_map = assign_cluster_names(df)
    df["cluster_naam"] = df["cluster_id"].map(label_map)

    return df, features, X_scaled_df, kmeans, pca, centroids_pca, (ks, inertias, silhouettes), label_map


def create_visualization(
    df: pd.DataFrame,
    pca: PCA,
    centroids_pca: np.ndarray,
    eval_metrics: tuple,
    label_map: dict,
    out_path: Path,
):
    print(f"[4/4] Genereren van visualisatie -> {out_path.name}...")
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.5), dpi=160)
    fig.suptitle(
        f"TMDB Movies Dataset — Data Wrangling & Euclidische K-Means Clustering (n = {len(df):,}, k = 5)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    palette = {
        "Blockbusters & Mega-Hits": "#e63946",
        "Mid-Budget Mainstream": "#457b9d",
        "Hooggewaardeerde Drama & Prestige": "#7b2cbf",
        "Low-Budget / Indie Hits": "#2a9d8f",
        "Box-Office Flops": "#f4a261",
    }
    hue_order = list(palette.keys())

    # --- Paneel 1: Elbow (WCSS / Inertia) & Euclidische Silhouette ---
    ax1 = axes[0, 0]
    ks, inertias, silhouettes = eval_metrics
    color_inertia = "#1d3557"
    color_sil = "#e76f51"

    l1 = ax1.plot(ks, inertias, marker="o", linewidth=2.2, color=color_inertia, label="Inertia / WCSS (Elbow)")
    ax1.axvline(5, linestyle="--", color="#2b9348", linewidth=1.8, label="Gekozen k = 5 (Silhouette-piek)")
    ax1.set_xlabel("Aantal clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color_inertia, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_inertia)
    ax1.set_xticks(ks)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(
        ks,
        silhouettes,
        marker="s",
        linewidth=2.2,
        linestyle="-.",
        color=color_sil,
        label="Silhouette Score (Euclidisch)",
    )
    ax1_twin.set_ylabel("Silhouette Score (L2)", color=color_sil, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_sil)
    ax1_twin.grid(False)

    lines = l1 + l2 + [ax1.lines[1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)
    ax1.set_title("1. Bepaling aantal clusters: Inertia (WCSS) & Silhouette", fontsize=12.5, fontweight="bold")

    # --- Paneel 2: PCA 2D Cluster Projectie ---
    ax2 = axes[0, 1]
    sns.scatterplot(
        data=df,
        x="pc1",
        y="pc2",
        hue="cluster_naam",
        hue_order=hue_order,
        palette=palette,
        alpha=0.45,
        s=20,
        edgecolor="none",
        ax=ax2,
    )
    for cid, (cx, cy) in enumerate(centroids_pca):
        cname = label_map[cid]
        ax2.scatter(
            cx,
            cy,
            s=240,
            c=palette[cname],
            marker="X",
            edgecolors="black",
            linewidths=1.8,
            zorder=10,
        )

    var1, var2 = pca.explained_variance_ratio_ * 100
    ax2.set_xlabel(f"Principal Component 1 ({var1:.1f}% verklaarde variantie)", fontweight="bold")
    ax2.set_ylabel(f"Principal Component 2 ({var2:.1f}% verklaarde variantie)", fontweight="bold")
    ax2.set_title("2. K-Means Clusters in 2D PCA-ruimte (X = Centroïde)", fontsize=12.5, fontweight="bold")
    ax2.legend(title="Cluster", loc="upper left", fontsize=8.5, title_fontsize=9.5, frameon=True)

    # --- Paneel 3: Budget vs. Revenue (Log-schaal) ---
    ax3 = axes[1, 0]
    sns.scatterplot(
        data=df,
        x=df["budget"] / 1e6,
        y=df["revenue"] / 1e6,
        hue="cluster_naam",
        hue_order=hue_order,
        palette=palette,
        alpha=0.45,
        s=20,
        edgecolor="none",
        ax=ax3,
        legend=False,
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    lims = [0.01, 2000]
    ax3.plot(lims, lims, "--", color="#222222", linewidth=1.4, label="Break-even (Revenue = Budget)")
    ax3.set_xlim(lims)
    ax3.set_ylim([0.01, 3500])
    ax3.set_xlabel("Budget in miljoenen USD (log-schaal)", fontweight="bold")
    ax3.set_ylabel("Revenue in miljoenen USD (log-schaal)", fontweight="bold")
    ax3.set_title("3. Financieel profiel: Budget vs. Opbrengst per Cluster", fontsize=12.5, fontweight="bold")
    ax3.legend(loc="lower right", frameon=True)

    offsets = {
        "Blockbusters & Mega-Hits": (-55, 8),
        "Mid-Budget Mainstream": (8, -12),
        "Hooggewaardeerde Drama & Prestige": (-70, 8),
        "Low-Budget / Indie Hits": (-55, 8),
        "Box-Office Flops": (-45, -14),
    }
    for cname in hue_order:
        sub = df[df["cluster_naam"] == cname].sort_values("vote_count", ascending=False)
        if not sub.empty:
            top_movie = sub.iloc[0]
            ax3.annotate(
                top_movie["title"][:22],
                xy=(top_movie["budget"] / 1e6, top_movie["revenue"] / 1e6),
                xytext=offsets.get(cname, (6, 6)),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=palette[cname], alpha=0.9),
            )

    # --- Paneel 4: Gestandaardiseerd Cluster-profiel (Heatmap van Centroïde Z-scores) ---
    ax4 = axes[1, 1]
    profile_cols = [
        "log_budget",
        "log_revenue",
        "log_roi",
        "runtime",
        "vote_average",
        "log_vote_count",
        "log_popularity",
    ]
    pretty_names = [
        "Budget (log)",
        "Revenue (log)",
        "ROI / Rendement",
        "Speelduur",
        "Beoordeling",
        "Aantal stemmen",
        "Populariteit",
    ]
    z_df = df.copy()
    z_df[profile_cols] = (z_df[profile_cols] - z_df[profile_cols].mean()) / z_df[profile_cols].std()
    prof = z_df.groupby("cluster_naam")[profile_cols].mean().reindex(hue_order)
    prof.columns = pretty_names

    sns.heatmap(
        prof,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Gemiddelde Z-score (Centroïde)"},
        linewidths=0.8,
        ax=ax4,
    )
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=25, ha="right")
    ax4.set_title("4. Cluster-centroïden (Gemiddelde Z-score per kenmerk)", fontsize=12.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    out_path.write_bytes(buf.getvalue())
    plt.close(fig)


def main():
    df = load_and_wrangle(INPUT_CSV)
    df, features, X_scaled_df, kmeans, pca, centroids_pca, eval_metrics, label_map = run_kmeans(
        df, k_final=5
    )

    print("\n=== CLUSTER SAMENVATTING (k = 5, Euclidische L2-afstand) ===")
    summary_table = (
        df.groupby("cluster_naam")
        .agg(
            aantal=("id", "count"),
            mediaan_budget_M=("budget", lambda s: round(s.median() / 1e6, 2)),
            mediaan_revenue_M=("revenue", lambda s: round(s.median() / 1e6, 2)),
            mediaan_roi=("roi_ratio", lambda s: round(s.median(), 2)),
            gem_rating=("vote_average", lambda s: round(s.mean(), 2)),
            gem_runtime=("runtime", lambda s: round(s.mean(), 1)),
            gem_l2_afstand=("centroid_l2_distance", lambda s: round(s.mean(), 2)),
            top_genre=("primary_genre", lambda s: s.mode().iloc[0]),
        )
        .sort_values("mediaan_budget_M", ascending=False)
    )
    print(summary_table.to_string())

    OUTPUT_CSV.write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"\nOpgeschoonde dataset met clusters opgeslagen in: {OUTPUT_CSV.name}")

    create_visualization(df, pca, centroids_pca, eval_metrics, label_map, OUTPUT_PLOT)
    print(f"Visualisatie opgeslagen in: {OUTPUT_PLOT.name}")


if __name__ == "__main__":
    main()
