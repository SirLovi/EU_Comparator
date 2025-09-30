"""Streamlit application to compare European Parliament member votes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import inspect
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast
import datetime as dt
import io
import zipfile

import pandas as pd
import streamlit as st
import altair as alt

DATA_DIR = Path(__file__).resolve().parent / "DATA"


_ALTAIR_SUPPORTS_WIDTH = "width" in inspect.signature(st.altair_chart).parameters


@dataclass(frozen=True)
class EntitySpec:
    """Describe a selectable comparison entity."""

    kind: str
    identifier: str


ENTITY_POSITION_PRIORITY = ("FOR", "AGAINST", "ABSTENTION", "DID_NOT_VOTE")
MIXED_POSITION = "MIXED"
STYLER_CELL_LIMIT = 250_000


def _render_full_width_altair(chart: alt.Chart, **kwargs: Any) -> Any:
    """Render Altair charts stretched to the container width across Streamlit versions."""
    if _ALTAIR_SUPPORTS_WIDTH:
        altair_chart = cast(Callable[..., Any], st.altair_chart)
        return altair_chart(chart, width="stretch", **kwargs)
    return st.altair_chart(chart, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def load_members() -> pd.DataFrame:
    """Load members with a handy full name column."""
    members = pd.read_csv(DATA_DIR / "members.csv.gz")
    members["country_code"] = members["country_code"].fillna("?")
    members["full_name"] = (
        members[["first_name", "last_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        + " ("
        + members["country_code"]
        + ")"
    )
    members = members.sort_values("full_name").reset_index(drop=True)
    return members


@st.cache_data(show_spinner=False)
def load_votes() -> pd.DataFrame:
    """Load vote metadata."""
    votes = pd.read_csv(DATA_DIR / "votes.csv.gz", parse_dates=["timestamp"])
    columns = [
        "id",
        "timestamp",
        "display_title",
        "procedure_reference",
        "procedure_title",
        "description",
        "is_main",
    ]
    existing_columns = [column for column in columns if column in votes.columns]
    return votes[existing_columns]


@st.cache_data(show_spinner=False)
def load_member_votes() -> pd.DataFrame:
    """Load the vote positions of all members."""
    return pd.read_csv(
        DATA_DIR / "member_votes.csv.gz",
        usecols=["vote_id", "member_id", "position", "group_code", "country_code"],
        dtype={
            "vote_id": "int64",
            "member_id": "int64",
            "position": "category",
            "group_code": "string",
            "country_code": "string",
        },
    )


@st.cache_data(show_spinner=False)
def load_vote_subjects() -> pd.DataFrame:
    """Load OEIL subject labels per vote with aggregated representations."""
    subjects = pd.read_csv(
        DATA_DIR / "oeil_subjects.csv.gz",
        dtype={"code": "string"},
    )
    subjects["code"] = subjects["code"].fillna("")
    subjects["top_code"] = subjects["code"].str.split(".").str[0]

    top_label_map = (
        subjects.drop_duplicates("code").set_index("code")["label"].to_dict()
    )
    subjects["top_label"] = subjects["top_code"].map(top_label_map)
    subjects["top_label"] = subjects.apply(
        lambda row: (
            row["top_label"]
            if isinstance(row["top_label"], str) and row["top_label"].strip()
            else row["label"]
        ),
        axis=1,
    )

    subject_votes = pd.read_csv(
        DATA_DIR / "oeil_subject_votes.csv.gz",
        dtype={"oeil_subject_code": "string"},
    )

    merged = subject_votes.merge(
        subjects[["code", "label", "top_label"]],
        left_on="oeil_subject_code",
        right_on="code",
        how="left",
    )

    def aggregate(values: pd.Series) -> tuple[str, ...]:
        unique = sorted({v for v in values if isinstance(v, str) and v.strip()})
        return tuple(unique)

    aggregated = (
        merged.groupby("vote_id")
        .agg(
            subjects=("label", aggregate),
            top_subjects=("top_label", aggregate),
        )
        .reset_index()
    )

    aggregated["subjects_display"] = aggregated["subjects"].apply(
        lambda values: ", ".join(values) if values else "—"
    )
    aggregated["top_subjects_display"] = aggregated["top_subjects"].apply(
        lambda values: ", ".join(values) if values else "—"
    )

    return aggregated


@st.cache_data(show_spinner=False)
def load_groups() -> pd.DataFrame:
    """Load political groups with friendly labels."""
    groups = pd.read_csv(DATA_DIR / "groups.csv.gz")
    columns = ["code", "label", "short_label", "official_label"]
    for column in columns:
        if column not in groups.columns:
            groups[column] = None
    groups["display"] = groups.apply(
        lambda row: (
            row["label"]
            if isinstance(row["label"], str) and row["label"].strip()
            else row.get("official_label") or row.get("short_label") or row.get("code")
        ),
        axis=1,
    )
    groups["display"] = groups["display"].fillna(groups["code"])
    return groups


@st.cache_data(show_spinner=False)
def load_countries() -> pd.DataFrame:
    """Load EU countries with ISO codes and labels."""
    countries = pd.read_csv(DATA_DIR / "countries.csv.gz")
    for column in ("code", "iso_alpha_2", "label"):
        if column not in countries.columns:
            countries[column] = None
    countries["display"] = countries.apply(
        lambda row: row["label"] or row["code"], axis=1
    )
    return countries


@st.cache_data(show_spinner=False)
def load_vote_geographies() -> pd.DataFrame:
    """Aggregate geographic focus metadata per vote."""
    geo_votes = pd.read_csv(
        DATA_DIR / "geo_area_votes.csv.gz",
        dtype={"geo_area_code": "string", "vote_id": "int64"},
    )
    if geo_votes.empty:
        return pd.DataFrame(columns=["vote_id", "geo_areas", "geo_areas_display"])

    geo_areas = pd.read_csv(
        DATA_DIR / "geo_areas.csv.gz",
        dtype={"code": "string", "label": "string", "iso_alpha_2": "string"},
    )
    geo_lookup = geo_areas.set_index("code")["label"].fillna("")

    merged = geo_votes.merge(
        geo_lookup.rename("geo_label"),
        left_on="geo_area_code",
        right_index=True,
        how="left",
    )

    def aggregate(values: pd.Series) -> tuple[str, ...]:
        unique = sorted({v for v in values if isinstance(v, str) and v.strip()})
        return tuple(unique)

    aggregated = (
        merged.groupby("vote_id")
        .agg(
            geo_areas=("geo_label", aggregate),
        )
        .reset_index()
    )

    aggregated["geo_areas_display"] = aggregated["geo_areas"].apply(
        lambda values: ", ".join(values) if values else "—"
    )
    return aggregated


@st.cache_data(show_spinner=False)
def load_vote_eurovoc() -> pd.DataFrame:
    """Aggregate EuroVoc concepts associated with each vote."""
    concept_votes = pd.read_csv(
        DATA_DIR / "eurovoc_concept_votes.csv.gz",
        dtype={"eurovoc_concept_id": "string", "vote_id": "int64"},
    )
    if concept_votes.empty:
        return pd.DataFrame(
            columns=["vote_id", "eurovoc_concepts", "eurovoc_concepts_display"]
        )

    concepts = pd.read_csv(
        DATA_DIR / "eurovoc_concepts.csv.gz",
        dtype={"id": "string", "label": "string"},
    )

    merged = concept_votes.merge(
        concepts.set_index("id")["label"].rename("concept_label"),
        left_on="eurovoc_concept_id",
        right_index=True,
        how="left",
    )

    def aggregate(values: pd.Series) -> tuple[str, ...]:
        unique = sorted({v for v in values if isinstance(v, str) and v.strip()})
        return tuple(unique)

    aggregated = (
        merged.groupby("vote_id")
        .agg(
            eurovoc_concepts=("concept_label", aggregate),
        )
        .reset_index()
    )

    aggregated["eurovoc_concepts_display"] = aggregated["eurovoc_concepts"].apply(
        lambda values: ", ".join(values) if values else "—"
    )
    return aggregated


@lru_cache(maxsize=1)
def member_lookup() -> Dict[int, str]:
    members = load_members()
    return dict(zip(members["id"], members["full_name"]))


@lru_cache(maxsize=1)
def group_lookup() -> Dict[str, str]:
    groups = load_groups()
    return dict(zip(groups["code"], groups["display"]))


@lru_cache(maxsize=1)
def country_lookup() -> Dict[str, str]:
    countries = load_countries()
    return dict(zip(countries["code"], countries["display"]))


def _to_category_list(value: object) -> List[str]:
    """Normalise stored category tuples to a list of readable labels."""
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _aggregate_entity_positions(subset: pd.DataFrame) -> pd.DataFrame:
    """Summarise vote positions for an aggregate entity (group or country)."""
    if subset.empty:
        return pd.DataFrame(columns=["vote_id", "position", "total_votes", "top_share"])

    counts = (
        subset.groupby(["vote_id", "position"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ENTITY_POSITION_PRIORITY, fill_value=0)
    )

    total = counts.sum(axis=1)
    top_counts = counts.max(axis=1)
    tie_mask = counts.eq(top_counts, axis=0).sum(axis=1) > 1
    top_position = counts.idxmax(axis=1)

    result = pd.DataFrame(
        {
            "vote_id": counts.index,
            "position": top_position,
            "total_votes": total.astype(int),
            "top_share": (top_counts / total).fillna(0.0),
        }
    )

    result.loc[result["total_votes"] == 0, "position"] = None
    result.loc[tie_mask, "position"] = MIXED_POSITION
    return result.reset_index(drop=True)


def _entity_display_name(spec: EntitySpec) -> str:
    """Construct a readable label for a selected entity."""
    if spec.kind == "member":
        try:
            member_id = int(spec.identifier)
        except (TypeError, ValueError):
            return f"Member {spec.identifier}"
        return member_lookup().get(member_id, f"Member {member_id}")
    if spec.kind == "group":
        label = group_lookup().get(spec.identifier, spec.identifier)
        return f"Group • {label}"
    if spec.kind == "country":
        label = country_lookup().get(spec.identifier, spec.identifier)
        return f"Country • {label}"
    return spec.identifier


def _entity_series(
    member_votes: pd.DataFrame, spec: EntitySpec
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Return the vote series and metadata for a given entity."""
    meta: Dict[str, Any] = {"kind": spec.kind, "identifier": spec.identifier}
    if spec.kind == "member":
        try:
            member_id = int(spec.identifier)
        except (TypeError, ValueError):
            member_id = None
        if member_id is None:
            return pd.Series(dtype="object"), meta
        subset = member_votes[member_votes["member_id"] == member_id]
        series = subset.set_index("vote_id")["position"].astype("string")
        return series, meta

    if spec.kind == "group":
        subset = member_votes[member_votes["group_code"] == spec.identifier]
        aggregated = _aggregate_entity_positions(subset)
        meta["aggregate"] = aggregated.set_index("vote_id").to_dict("index")
        return aggregated.set_index("vote_id")["position"].astype("string"), meta

    if spec.kind == "country":
        subset = member_votes[member_votes["country_code"] == spec.identifier]
        aggregated = _aggregate_entity_positions(subset)
        meta["aggregate"] = aggregated.set_index("vote_id").to_dict("index")
        return aggregated.set_index("vote_id")["position"].astype("string"), meta

    return pd.Series(dtype="object"), meta


@st.cache_data(show_spinner=False)
def load_filtered_votes(
    entities: Tuple[EntitySpec, ...],
    main_only: bool,
    categories: Tuple[str, ...],
    geographies: Tuple[str, ...],
    date_range: Tuple[Optional[str], Optional[str]],
) -> pd.DataFrame:
    """Return cached vote matrix filtered by vote type, geography, and dates."""
    base = build_vote_matrix(list(entities))
    if base.empty:
        return base

    filtered = base.copy()

    if main_only:
        filtered = filtered[filtered["is_main"]]

    if categories:
        category_set = set(categories)
        filtered = filtered[
            filtered["top_subjects"].apply(
                lambda cats: bool(set(_to_category_list(cats)) & category_set)
            )
        ]

    if geographies:
        geography_set = set(geographies)
        filtered = filtered[
            filtered["geo_areas"].apply(
                lambda cats: bool(set(_to_category_list(cats)) & geography_set)
            )
        ]

    start_iso, end_iso = date_range
    if start_iso:
        start_dt = pd.to_datetime(start_iso)
        filtered = filtered[filtered["timestamp"] >= start_dt]
    if end_iso:
        end_dt = pd.to_datetime(end_iso) + pd.Timedelta(days=1)
        filtered = filtered[filtered["timestamp"] < end_dt]

    filtered.attrs = base.attrs
    result = filtered.reset_index(drop=True)
    result.attrs = filtered.attrs
    return result


PAIR_DELIMITER = "||"


def encode_pair_token(left: str, right: str) -> str:
    return f"{left}{PAIR_DELIMITER}{right}"


def decode_pair_token(token: str) -> Tuple[str, str] | None:
    if not token or PAIR_DELIMITER not in token:
        return None
    left, right = token.split(PAIR_DELIMITER, 1)
    if not left or not right:
        return None
    return left, right


def parse_search_query(query: Optional[str]) -> List[List[str]]:
    """Parse a simple query string into groups of AND terms combined with OR."""
    if not query:
        return []

    stripped = query.strip()
    if not stripped:
        return []

    groups = re.split(r"\s+OR\s+", stripped, flags=re.IGNORECASE)
    parsed: List[List[str]] = []
    for group in groups:
        terms = [term.lower() for term in group.strip().split() if term.strip()]
        if terms:
            parsed.append(terms)
    return parsed


def build_search_mask(
    df: pd.DataFrame, query_groups: List[List[str]], columns: List[str]
) -> pd.Series:
    """Return a boolean mask matching rows that satisfy the query groups."""
    if not query_groups or not columns:
        return pd.Series(True, index=df.index)

    def normalise(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value if item)
        return str(value)

    combined = pd.Series("", index=df.index, dtype="string")
    for column in columns:
        if column not in df.columns:
            continue
        column_values = df[column].apply(normalise)
        combined = (combined.fillna("") + " " + column_values.fillna("")).str.strip()

    combined = combined.str.lower()
    mask = pd.Series(False, index=df.index)
    for group in query_groups:
        group_mask = pd.Series(True, index=df.index)
        for term in group:
            group_mask &= combined.str.contains(re.escape(term), na=False)
        mask |= group_mask
    return mask


def build_vote_matrix(selected_entities: List[EntitySpec]) -> pd.DataFrame:
    """Create a wide table of common votes for the selected entities."""
    if not selected_entities:
        return pd.DataFrame()

    member_votes = load_member_votes()

    entity_series_list: List[pd.Series] = []
    entity_names: List[str] = []
    entity_metadata: Dict[str, Dict[str, Any]] = {}
    name_counts: Dict[str, int] = {}

    for spec in selected_entities:
        series, meta = _entity_series(member_votes, spec)
        if series.empty:
            continue

        base_name = _entity_display_name(spec)
        suffix = name_counts.get(base_name, 0)
        if suffix:
            display_name = f"{base_name} ({suffix + 1})"
        else:
            display_name = base_name
        name_counts[base_name] = suffix + 1

        entity_series_list.append(series)
        entity_names.append(display_name)
        entity_metadata[display_name] = meta

    if not entity_series_list:
        return pd.DataFrame()

    common_vote_ids = set(entity_series_list[0].index)
    for series in entity_series_list[1:]:
        common_vote_ids &= set(series.index)

    if not common_vote_ids:
        return pd.DataFrame()

    ordered_vote_ids = sorted(common_vote_ids)
    pivot_data = {}
    for name, series in zip(entity_names, entity_series_list):
        pivot_data[name] = series.reindex(ordered_vote_ids)

    pivot = pd.DataFrame(pivot_data, index=ordered_vote_ids)

    vote_details = load_votes().set_index("id")
    pivot = pivot.join(vote_details, how="left")

    def summarise(row: pd.Series) -> pd.Series:
        entity_positions = row[entity_names].dropna()
        unique_positions = set(entity_positions.tolist())
        all_same = len(unique_positions) == 1
        shared_position = entity_positions.iloc[0] if all_same else None
        return pd.Series(
            {
                "agreement": "Same" if all_same else "Different",
                "shared_position": shared_position,
            }
        )

    summary = pivot.apply(summarise, axis=1)
    combined = pd.concat([pivot, summary], axis=1)
    combined = combined.reset_index().rename(columns={"index": "vote_id"})

    categories = load_vote_subjects()
    combined = combined.merge(categories, on="vote_id", how="left")

    geographies = load_vote_geographies()
    if not geographies.empty:
        combined = combined.merge(geographies, on="vote_id", how="left")

    eurovoc = load_vote_eurovoc()
    if not eurovoc.empty:
        combined = combined.merge(eurovoc, on="vote_id", how="left")

    def ensure_tuple(value: object) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return tuple()

    tuple_columns = [
        "subjects",
        "top_subjects",
        "geo_areas",
        "eurovoc_concepts",
    ]
    for column in tuple_columns:
        if column in combined.columns:
            combined[column] = combined[column].apply(ensure_tuple)

    display_columns = [
        "subjects_display",
        "top_subjects_display",
        "geo_areas_display",
        "eurovoc_concepts_display",
    ]
    for column in display_columns:
        if column in combined.columns:
            combined[column] = combined[column].fillna("—")

    combined = combined.sort_values("timestamp", ascending=False)
    combined.attrs["entity_names"] = entity_names
    combined.attrs["entity_metadata"] = entity_metadata
    return combined


def format_vote_table(df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """Select, rename, and prettify columns for display & download."""
    if df.empty:
        return df

    base_columns = [
        "timestamp",
        "display_title",
        "procedure_reference",
        "procedure_title",
        "is_main",
        "top_subjects_display",
        "subjects_display",
        "agreement",
        "shared_position",
    ]

    optional_columns: List[str] = []
    if "geo_areas_display" in df.columns:
        optional_columns.append("geo_areas_display")
    if "eurovoc_concepts_display" in df.columns:
        optional_columns.append("eurovoc_concepts_display")

    for column in optional_columns:
        if column not in df.columns:
            df[column] = "—"

    display_df = df[[*base_columns, *optional_columns, *selected_columns]].copy()

    # Pretty timestamps
    if not pd.api.types.is_datetime64_any_dtype(display_df["timestamp"]):
        display_df["timestamp"] = pd.to_datetime(
            display_df["timestamp"], errors="coerce", utc=True
        )
    if (
        hasattr(display_df["timestamp"].dtype, "tz")
        and display_df["timestamp"].dt.tz is not None
    ):
        display_df["timestamp"] = display_df["timestamp"].dt.tz_convert(None)
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    # Rename headers
    display_df = display_df.rename(
        columns={
            "timestamp": "Vote time",
            "display_title": "Vote title",
            "procedure_reference": "Procedure ref",
            "procedure_title": "Procedure title",
            "is_main": "Main vote",
            "top_subjects_display": "Top categories",
            "subjects_display": "Categories",
            "geo_areas_display": "Geographical focus",
            "eurovoc_concepts_display": "EuroVoc concepts",
            "agreement": "Agreement",
            "shared_position": "Shared position",
        }
    )

    # Human-friendly values
    display_df["Main vote"] = (
        display_df["Main vote"].map({True: "Yes", False: "No"}).fillna("—")
    )
    display_df["Shared position"] = display_df["Shared position"].fillna("—")
    display_df["Top categories"] = display_df["Top categories"].fillna("—")
    display_df["Categories"] = display_df["Categories"].fillna("—")
    if "Geographical focus" in display_df.columns:
        display_df["Geographical focus"] = display_df["Geographical focus"].fillna("—")
    if "EuroVoc concepts" in display_df.columns:
        display_df["EuroVoc concepts"] = display_df["EuroVoc concepts"].fillna("—")

    return display_df


def style_vote_table(
    display_df: pd.DataFrame,
) -> pd.io.formats.style.Styler | pd.DataFrame:
    """Apply colour styling to highlight agreement and vote positions.
    Returns a Styler if styling is applicable; otherwise the raw DataFrame.
    """
    if display_df.empty:
        return display_df

    if display_df.size > STYLER_CELL_LIMIT:
        display_df.attrs["styling_skipped"] = True
        return display_df

    position_styles: dict[str | None, str] = {
        "FOR": "background-color: #d1e7dd; color: #0f5132; font-weight: 600;",
        "AGAINST": "background-color: #f8d7da; color: #842029; font-weight: 600;",
        "ABSTENTION": "background-color: #fff3cd; color: #664d03; font-weight: 600;",
        "DID_NOT_VOTE": "background-color: #e2e3e5; color: #41464b;",
        "MIXED": "background-color: #e2e3e5; color: #212529; font-style: italic;",
        "—": "color: #6c757d;",
        None: "color: #6c757d;",
    }

    def agreement_style(value: object) -> str:
        if value == "Same":
            return "background-color: #d1e7dd; color: #0f5132; font-weight: 700;"
        if value == "Different":
            return "background-color: #f8d7da; color: #842029; font-weight: 700;"
        return ""

    def position_style(value: object) -> str:
        if value is None:
            return position_styles.get(None, "")
        if isinstance(value, str):
            return position_styles.get(value, "")
        return ""

    base_columns = {
        "Vote time",
        "Vote title",
        "Procedure ref",
        "Procedure title",
        "Main vote",
        "Top categories",
        "Categories",
        "Geographical focus",
        "EuroVoc concepts",
        "Agreement",
        "Shared position",
    }
    position_columns = [col for col in display_df.columns if col not in base_columns]

    styler = display_df.style
    styler = styler.map(agreement_style, subset=["Agreement"])
    styler = styler.map(position_style, subset=["Shared position", *position_columns])
    styler = styler.set_properties(**{"white-space": "nowrap"})  # type: ignore[arg-type]
    return styler


def pairwise_agreement(df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """Return a summary of how often each pair of entities agreed."""
    if df.empty or len(selected_columns) < 2:
        return pd.DataFrame()

    rows = []
    total_votes = len(df)

    for left, right in combinations(selected_columns, 2):
        same_mask = df[left] == df[right]
        same_count = int(same_mask.sum())
        different_count = int(total_votes - same_count)
        agreement_rate = same_count / total_votes if total_votes else float("nan")
        rows.append(
            {
                "Pair": f"{left} ↔ {right}",
                "Same": same_count,
                "Different": different_count,
                "Agreement rate": f"{agreement_rate:.0%}" if total_votes else "N/A",
            }
        )

    return pd.DataFrame(rows)


def pairwise_trend(
    df: pd.DataFrame,
    left: str,
    right: str,
    frequency: str = "M",
) -> pd.DataFrame:
    """Summarise agreement between two entities across time buckets."""
    if df.empty or left not in df.columns or right not in df.columns:
        return pd.DataFrame()

    subset = df[["timestamp", left, right]].dropna().copy()
    if subset.empty:
        return pd.DataFrame()

    subset["timestamp"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.dropna(subset=["timestamp"])
    if subset.empty:
        return pd.DataFrame()

    if subset["timestamp"].dt.tz is not None:
        subset["timestamp"] = subset["timestamp"].dt.tz_convert(None)

    subset["same"] = (subset[left] == subset[right]).astype(int)
    subset["period"] = subset["timestamp"].dt.to_period(frequency).dt.to_timestamp()

    grouped = (
        subset.groupby("period")
        .agg(total_votes=("timestamp", "count"), same_votes=("same", "sum"))
        .reset_index()
        .sort_values("period")
    )

    if grouped.empty:
        return pd.DataFrame()

    grouped["agreement_rate"] = (grouped["same_votes"] / grouped["total_votes"]).fillna(
        0.0
    )
    grouped["left"] = left
    grouped["right"] = right
    return grouped


def pairwise_trend_chart(trend_df: pd.DataFrame) -> alt.Chart | None:
    if trend_df.empty:
        return None

    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("period:T", title="Vote month"),
            y=alt.Y(
                "agreement_rate:Q",
                title="Agreement rate",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("period:T", title="Period"),
                alt.Tooltip(
                    "agreement_rate:Q",
                    title="Agreement",
                    format=".0%",
                ),
                alt.Tooltip("same_votes:Q", title="Same votes"),
                alt.Tooltip("total_votes:Q", title="Total votes"),
            ],
        )
        .properties(height=260, width="container")
    )
    return chart


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise filtered votes by their top-level OEIL subjects."""
    if df.empty or "top_subjects" not in df.columns:
        return pd.DataFrame()

    subject_lists = df["top_subjects"].apply(_to_category_list)
    mask = subject_lists.map(bool)
    if not mask.any():
        return pd.DataFrame()

    exploded = (
        df.loc[mask, ["vote_id", "agreement"]]
        .assign(top_subjects=subject_lists[mask])
        .explode("top_subjects")
    )

    if exploded.empty:
        return pd.DataFrame()

    summary = (
        exploded.groupby("top_subjects")
        .agg(
            Votes=("vote_id", "nunique"),
            Same=("agreement", lambda s: int((s == "Same").sum())),
        )
        .reset_index()
    )

    if summary.empty:
        return pd.DataFrame()

    summary["Different"] = summary["Votes"] - summary["Same"]
    summary["Agreement rate"] = summary.apply(
        lambda row: f"{row['Same'] / row['Votes']:.0%}" if row["Votes"] else "N/A",
        axis=1,
    )

    summary = summary.rename(columns={"top_subjects": "Top-level subject"})
    summary = summary.sort_values(
        ["Votes", "Top-level subject"], ascending=[False, True]
    )
    return summary.reset_index(drop=True)


def pairwise_agreement_heatmap(
    df: pd.DataFrame, selected_columns: List[str]
) -> alt.Chart | None:
    """Return a heatmap chart of pairwise agreement rates."""
    if df.empty or len(selected_columns) < 2:
        return None

    matrix = pd.DataFrame(
        1.0, index=selected_columns, columns=selected_columns, dtype=float
    )

    for left, right in combinations(selected_columns, 2):
        subset = df[[left, right]].dropna()
        total = len(subset)
        if total:
            same = (subset[left] == subset[right]).sum()
            rate = same / total
        else:
            rate = float("nan")
        matrix.loc[left, right] = matrix.loc[right, left] = rate

    matrix = matrix.fillna(0.0)
    for name in selected_columns:
        matrix.loc[name, name] = 1.0

    heatmap_data = matrix.reset_index().melt(
        id_vars="index", var_name="Entity B", value_name="Agreement"
    )
    heatmap_data = heatmap_data.rename(columns={"index": "Entity A"})

    chart = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X("Entity B:N", sort=selected_columns, title=""),
            y=alt.Y("Entity A:N", sort=selected_columns, title=""),
            color=alt.Color(
                "Agreement:Q",
                scale=alt.Scale(domain=[0, 1], scheme="blues"),
                legend=alt.Legend(title="Agreement"),
            ),
            tooltip=[
                alt.Tooltip("Entity A", title="Entity A"),
                alt.Tooltip("Entity B", title="Entity B"),
                alt.Tooltip(
                    "Agreement",
                    title="Agreement",
                    format=".0%",
                ),
            ],
        )
        .properties(height=220, width=220)
    )

    return chart


def category_agreement_chart(breakdown: pd.DataFrame) -> alt.Chart | None:
    """Return a stacked bar chart highlighting agreement per subject."""
    if breakdown.empty:
        return None

    chart_data = breakdown.melt(
        id_vars=["Top-level subject", "Agreement rate", "Votes"],
        value_vars=["Same", "Different"],
        var_name="Outcome",
        value_name="Count",
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            y=alt.Y("Top-level subject:N", sort="-x", title="Top-level subject"),
            x=alt.X("Count:Q", title="Share of votes", stack="normalize"),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Same", "Different"], range=["#198754", "#dc3545"]
                ),
                legend=alt.Legend(title="Outcome"),
            ),
            tooltip=[
                alt.Tooltip("Top-level subject:N", title="Subject"),
                alt.Tooltip("Outcome:N", title="Outcome"),
                alt.Tooltip("Count:Q", title="Votes"),
                alt.Tooltip("Votes:Q", title="Total"),
                alt.Tooltip("Agreement rate:N", title="Agreement rate"),
            ],
        )
        .properties(height=max(160, 24 * len(breakdown)), width="container")
    )

    return chart


def build_summary_metrics(
    total_votes: int, same_votes: int, different_votes: int
) -> pd.DataFrame:
    total = max(total_votes, 1)
    same_share = same_votes / total if total_votes else float("nan")
    different_share = different_votes / total if total_votes else float("nan")
    data = [
        {"Metric": "Shared votes analysed", "Value": total_votes, "Share": "—"},
        {
            "Metric": "Votes with identical positions",
            "Value": same_votes,
            "Share": f"{same_share:.0%}" if total_votes else "—",
        },
        {
            "Metric": "Votes with differing positions",
            "Value": different_votes,
            "Share": f"{different_share:.0%}" if total_votes else "—",
        },
    ]
    return pd.DataFrame(data)


def build_export_package(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("votes.csv", detail_df.to_csv(index=False))
        archive.writestr("summary.csv", summary_df.to_csv(index=False))
        if not breakdown_df.empty:
            archive.writestr("category_breakdown.csv", breakdown_df.to_csv(index=False))
        if not pairwise_df.empty:
            archive.writestr("pairwise_agreement.csv", pairwise_df.to_csv(index=False))

    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_last_updated() -> str:
    """Get the timestamp of the source dataset if available."""
    path = DATA_DIR / "last_updated.txt"
    if not path.exists():
        return "unknown"
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    st.set_page_config(
        page_title="EU Parliament Vote Comparator",
        page_icon="🇪🇺",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --eu-primary: #2841f5;
            --eu-soft: #f5f7ff;
            --eu-border: rgba(40, 65, 245, 0.22);
        }
        .stApp .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }
        .intro-card {
            background: linear-gradient(135deg, rgba(40, 65, 245, 0.08), rgba(40, 65, 245, 0.02));
            border-radius: 1rem;
            border: 1px solid var(--eu-border);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.5rem;
        }
        .intro-card p {
            margin-bottom: 0.5rem;
        }
        .intro-card ul {
            margin: 0;
            padding-left: 1.1rem;
        }
        .intro-card li {
            margin-bottom: 0.35rem;
        }
        .stApp div[data-testid="stTabs"] div[role="tablist"] {
            gap: 0.5rem;
        }
        .stApp div[data-testid="stTabs"] button[role="tab"] {
            background: var(--eu-soft);
            color: #111827;
            padding: 0.8rem 1.4rem;
            border-radius: 0.8rem;
            border: 1px solid var(--eu-border);
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.2s ease-in-out;
        }
        .stApp div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: var(--eu-primary);
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(40, 65, 245, 0.2);
            border-color: var(--eu-primary);
        }
        .stApp div[data-testid="stTabs"] button[role="tab"]:hover {
            border-color: var(--eu-primary);
            color: var(--eu-primary);
        }
        .stApp div[data-testid="stTabs"] div[data-baseweb="tab-content"] {
            background: #ffffff;
            border-radius: 1rem;
            border: 1px solid rgba(40, 65, 245, 0.1);
            padding: 1.5rem;
            margin-top: 0.75rem;
        }
        div[data-testid="metric-container"] {
            background: var(--eu-soft);
            border: 1px solid rgba(40, 65, 245, 0.1);
            border-radius: 0.8rem;
            padding: 1rem;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricLabel"] {
            color: #111827;
            font-weight: 600;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #111827;
        }
        @media (max-width: 768px) {
            .stApp div[data-testid="stTabs"] button[role="tab"] {
                font-size: 0.95rem;
                padding: 0.7rem 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("European Parliament Vote Comparator")
    st.caption(
        "Understand how Members of the European Parliament vote together across policy areas."
    )
    st.markdown(
        """
        <div class="intro-card">
            <p><strong>Make sense of shared roll-call votes in seconds.</strong></p>
            <ul>
                <li><strong>Select entities:</strong> mix individual Members, political groups, or national delegations.</li>
                <li><strong>Refine the focus:</strong> narrow by vote type, subjects, geography, and time period.</li>
                <li><strong>Explore the tabs:</strong> Subjects summarise topics, Agreement highlights alignment, and Votes lists every record.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    members = load_members()
    groups = load_groups()
    countries = load_countries()
    subjects_catalogue = load_vote_subjects()
    geography_catalogue = load_vote_geographies()

    top_subject_series = subjects_catalogue.get("top_subjects")
    top_category_options = (
        sorted(
            {cat for entry in top_subject_series for cat in _to_category_list(entry)}
        )
        if top_subject_series is not None
        else []
    )

    geo_area_series = (
        geography_catalogue.get("geo_areas") if not geography_catalogue.empty else None
    )
    geo_area_options = (
        sorted({area for entry in geo_area_series for area in _to_category_list(entry)})
        if geo_area_series is not None
        else []
    )

    params = st.query_params

    def param_list(key: str) -> List[str]:
        values = params.get_all(key)
        if values:
            return values
        value = params.get(key)
        if value is None:
            return []
        return [value]

    entity_options: Dict[str, Dict[str, Any]] = {}
    alias_to_key: Dict[str, str] = {}

    def register_entity(
        spec: EntitySpec, label: str, aliases: Iterable[Optional[str]]
    ) -> str:
        key = f"{spec.kind}:{spec.identifier}"
        entity_options[key] = {"label": label, "spec": spec}
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                alias_to_key[alias] = key
        return key

    member_option_pairs: List[Tuple[str, str]] = []
    for record in members.itertuples():
        spec = EntitySpec("member", str(record.id))
        full_name = str(getattr(record, "full_name", "")).strip()
        label = f"Member · {full_name}" if full_name else f"Member · {record.id}"
        member_aliases: List[str] = [full_name] if full_name else [str(record.id)]
        key = register_entity(spec, label, member_aliases)
        member_option_pairs.append((label, key))
    member_option_pairs.sort(key=lambda item: item[0])
    member_option_keys = [key for _, key in member_option_pairs]

    group_option_pairs: List[Tuple[str, str]] = []
    for record in groups.itertuples():
        spec = EntitySpec("group", str(record.code))
        display_label = str(getattr(record, "display", record.code) or "").strip()
        label = (
            f"Group · {display_label}" if display_label else f"Group · {record.code}"
        )
        group_aliases: List[str] = [str(record.code)]
        short_label = getattr(record, "short_label", None)
        if isinstance(short_label, str) and short_label.strip():
            group_aliases.append(short_label.strip())
        official_label = getattr(record, "official_label", None)
        if isinstance(official_label, str) and official_label.strip():
            group_aliases.append(official_label.strip())
        key = register_entity(spec, label, group_aliases)
        group_option_pairs.append((label, key))
    group_option_pairs.sort(key=lambda item: item[0])
    group_option_keys = [key for _, key in group_option_pairs]

    country_option_pairs: List[Tuple[str, str]] = []
    for record in countries.itertuples():
        spec = EntitySpec("country", str(record.code))
        display = str(getattr(record, "display", record.code) or "").strip()
        label = f"Country · {display}" if display else f"Country · {record.code}"
        country_aliases: List[str] = [str(record.code)]
        iso_alpha_2 = getattr(record, "iso_alpha_2", None)
        if isinstance(iso_alpha_2, str) and iso_alpha_2.strip():
            country_aliases.append(iso_alpha_2.strip())
        if display and display not in country_aliases:
            country_aliases.append(display)
        key = register_entity(spec, label, country_aliases)
        country_option_pairs.append((label, key))
    country_option_pairs.sort(key=lambda item: item[0])
    country_option_keys = [key for _, key in country_option_pairs]

    all_option_keys = [*member_option_keys, *group_option_keys, *country_option_keys]

    default_selected_keys = [
        key for key in param_list("entities") if key in entity_options
    ]
    if not default_selected_keys:
        for alias in param_list("members"):
            lookup_key: Optional[str] = alias_to_key.get(alias)
            if lookup_key:
                default_selected_keys.append(lookup_key)

    default_main_only = params.get("main_only", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    filter_options = ("All votes", "Only same", "Only different")
    filter_default = params.get("filter", filter_options[0])
    if filter_default not in filter_options:
        filter_default = filter_options[0]
    filter_option = filter_default

    category_search_default = params.get("subject_query", "")
    default_top_categories = [
        cat for cat in param_list("subjects") if cat in top_category_options
    ]

    geography_search_default = params.get("geo_query", "")
    default_geo_filters = [
        geo for geo in param_list("geographies") if geo in geo_area_options
    ]

    frequency_param = params.get("frequency", "M")
    allowed_frequencies = {"M": "Monthly", "Q": "Quarterly", "W": "Weekly"}
    if frequency_param not in allowed_frequencies:
        frequency_param = "M"

    date_start_param = params.get("date_start")
    date_end_param = params.get("date_end")
    try:
        default_date_start = (
            pd.to_datetime(date_start_param).date() if date_start_param else None
        )
    except Exception:  # pragma: no cover - defensive
        default_date_start = None
    try:
        default_date_end = (
            pd.to_datetime(date_end_param).date() if date_end_param else None
        )
    except Exception:  # pragma: no cover - defensive
        default_date_end = None

    focus_default = params.get("focus", "")
    vote_title_query_default = params.get("vote_query", "")

    st.sidebar.header("Choose entities")
    st.sidebar.markdown(
        "Mix individual Members, political groups, or national delegations."
    )

    if "entity_selection_override" in st.session_state:
        st.session_state["entity_selection"] = st.session_state.pop(
            "entity_selection_override"
        )

    st.session_state.setdefault("entity_selection", default_selected_keys)

    selected_keys: list[str] = st.sidebar.multiselect(
        "Entities to compare",
        options=all_option_keys,
        max_selections=20,
        key="entity_selection",
        format_func=lambda key: entity_options[key]["label"],
        help="Choose up to twenty entities to compare their voting records.",
    )

    def apply_quick_selection(keys: List[str]) -> None:
        valid = [key for key in keys if key in entity_options]
        missing = len(keys) - len(valid)
        if missing:
            st.warning("Some preset entries are unavailable and were skipped.")
        updated = list(dict.fromkeys([*selected_keys, *valid]))[:20]
        st.session_state["entity_selection_override"] = updated
        st.session_state["_rerun"] = not st.session_state.get("_rerun", False)
        st.rerun()

    quick_presets: Dict[str, List[str]] = {
        "Load Czech comparison": [
            alias_to_key.get(name, "")
            for name in [
                "Filip TUREK (CZE)",
                "Kateřina KONEČNÁ (CZE)",
                "Alexandr VONDRA (CZE)",
                "Jan FARSKÝ (CZE)",
                "Danuše NERUDOVÁ (CZE)",
                "Luděk NIEDERMAYER (CZE)",
            ]
        ],
        "Compare major political groups": [
            alias_to_key.get(code, "") for code in ["EPP", "S&D", "RE", "ID", "NI"]
        ],
        "Baltic delegations": [
            alias_to_key.get(code, "") for code in ["LTU", "LVA", "EST"]
        ],
    }

    for label, keys in quick_presets.items():
        preset_keys = [key for key in keys if key]
        if preset_keys and st.sidebar.button(label):
            apply_quick_selection(preset_keys)

    st.sidebar.caption("Compare up to twenty entities at once.")
    st.sidebar.divider()
    st.sidebar.subheader("Refine votes")
    st.sidebar.markdown(
        "Focus on specific vote types, subjects, places, or time windows."
    )
    main_only = st.sidebar.toggle(
        "Only include main votes",
        value=default_main_only,
        help=(
            "Main votes capture the decision on the final text. "
            "Disable to include amendments and procedural votes."
        ),
    )

    category_search = st.sidebar.text_input(
        "Search top-level subjects",
        value=category_search_default,
        help=("Filter the OEIL subject list. Clear the search to show every subject."),
    )

    if top_category_options:
        lowered = category_search.lower()
        filtered_category_options = (
            [cat for cat in top_category_options if lowered in cat.lower()]
            if category_search
            else top_category_options
        )
        available_category_options = sorted(
            set(filtered_category_options) | set(default_top_categories)
        )
    else:
        filtered_category_options = []
        available_category_options = sorted(set(default_top_categories))

    selected_top_categories = st.sidebar.multiselect(
        "Top-level subjects",
        options=available_category_options,
        default=default_top_categories,
        help=(
            "Filter the shared votes down to specific OEIL top-level subjects. "
            "Leave empty to keep every subject."
        ),
    )

    if top_category_options:
        st.sidebar.caption(
            f"{len(filtered_category_options)} of {len(top_category_options)} subjects match your search."
        )
    else:
        st.sidebar.caption("Subject metadata isn't available in this dataset release.")

    geography_search = st.sidebar.text_input(
        "Search geographic focus",
        value=geography_search_default,
        help="Look for votes tagged with specific countries or regions.",
    )

    if geo_area_options:
        geo_lower = geography_search.lower()
        filtered_geo_options = (
            [geo for geo in geo_area_options if geo_lower in geo.lower()]
            if geography_search
            else geo_area_options
        )
        available_geo_options = sorted(
            set(filtered_geo_options) | set(default_geo_filters)
        )
    else:
        filtered_geo_options = []
        available_geo_options = sorted(set(default_geo_filters))

    selected_geographies = st.sidebar.multiselect(
        "Geographic areas",
        options=available_geo_options,
        default=default_geo_filters,
        help=(
            "Limit the analysis to votes tagged with specific geographic areas. "
            "Leave empty to keep every area."
        ),
    )

    if geo_area_options:
        st.sidebar.caption(
            f"{len(filtered_geo_options)} of {len(geo_area_options)} areas match your search."
        )
    else:
        st.sidebar.caption(
            "Geographic metadata isn't available in this dataset release."
        )

    vote_timestamps = load_votes()["timestamp"]
    dataset_start = (
        pd.to_datetime(vote_timestamps.min())
        if not vote_timestamps.empty
        else pd.Timestamp.today()
    )
    dataset_end = (
        pd.to_datetime(vote_timestamps.max())
        if not vote_timestamps.empty
        else pd.Timestamp.today()
    )

    default_date_filter = bool(default_date_start or default_date_end)
    enable_date_filter = st.sidebar.toggle(
        "Filter by date range",
        value=default_date_filter,
        help="Restrict the comparison to votes within a custom period.",
    )

    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    if enable_date_filter:
        start_default = default_date_start or dataset_start.date()
        end_default = default_date_end or dataset_end.date()
        start_date_input = st.sidebar.date_input(
            "Start date",
            value=start_default,
            min_value=dataset_start.date(),
            max_value=dataset_end.date(),
        )
        end_date_input = st.sidebar.date_input(
            "End date",
            value=end_default,
            min_value=start_date_input,
            max_value=dataset_end.date(),
        )
        start_date = start_date_input
        end_date = end_date_input

    frequency_labels = list(allowed_frequencies.values())
    frequency_codes = list(allowed_frequencies.keys())
    default_frequency_index = frequency_codes.index(frequency_param)
    selected_frequency_label = st.sidebar.selectbox(
        "Trend aggregation",
        options=frequency_labels,
        index=default_frequency_index,
        help="Adjust the time bucket used in the agreement trend chart.",
    )
    trend_frequency = frequency_codes[frequency_labels.index(selected_frequency_label)]

    st.sidebar.caption(f"Dataset last updated • {load_last_updated()}")

    selected_entities = [
        entity_options[key]["spec"] for key in selected_keys if key in entity_options
    ]

    def sync_query_params(
        focus_value: str | None = None,
        pair_value: str | None = None,
        vote_query_value: str | None = None,
    ) -> None:
        payload: Dict[str, List[str]] = {
            "main_only": [str(main_only).lower()],
            "filter": [filter_option],
            "frequency": [trend_frequency],
        }
        if selected_keys:
            payload["entities"] = selected_keys
        if category_search:
            payload["subject_query"] = [category_search]
        if selected_top_categories:
            payload["subjects"] = selected_top_categories
        if geography_search:
            payload["geo_query"] = [geography_search]
        if selected_geographies:
            payload["geographies"] = selected_geographies
        if enable_date_filter and start_date:
            payload["date_start"] = [start_date.isoformat()]
        if enable_date_filter and end_date:
            payload["date_end"] = [end_date.isoformat()]
        if focus_value:
            payload["focus"] = [focus_value]
        if pair_value:
            payload["pair"] = [pair_value]
        if vote_query_value:
            payload["vote_query"] = [vote_query_value]

        candidate = {key: value for key, value in payload.items() if value}
        current = {key: params.get_all(key) for key in list(params.keys())}

        if current != candidate:
            params.clear()
            for key, value in candidate.items():
                params[key] = value if len(value) != 1 else value[0]

    if not selected_entities:
        sync_query_params()
        st.info("Select at least one entity to explore voting patterns.")
        return

    start_iso = start_date.isoformat() if start_date else None
    end_iso = end_date.isoformat() if end_date else None

    entity_tuple = tuple(selected_entities)
    category_tuple = tuple(selected_top_categories)
    geography_tuple = tuple(selected_geographies)
    vote_matrix = load_filtered_votes(
        entity_tuple,
        main_only,
        category_tuple,
        geography_tuple,
        (start_iso, end_iso),
    )

    if vote_matrix.empty:
        sync_query_params()
        st.warning(
            "No votes were found where all selected entities recorded a position. "
            "Try different selections or relax the filters."
        )
        return

    selected_columns = vote_matrix.attrs.get("entity_names", [])
    if not selected_columns:
        metadata_columns = {
            "vote_id",
            "timestamp",
            "display_title",
            "procedure_reference",
            "procedure_title",
            "is_main",
            "subjects",
            "subjects_display",
            "top_subjects",
            "top_subjects_display",
            "geo_areas",
            "geo_areas_display",
            "eurovoc_concepts",
            "eurovoc_concepts_display",
            "agreement",
            "shared_position",
        }
        selected_columns = [
            column for column in vote_matrix.columns if column not in metadata_columns
        ]

    same_votes = (vote_matrix["agreement"] == "Same").sum()
    different_votes = (vote_matrix["agreement"] == "Different").sum()

    total_votes = len(vote_matrix)
    same_share = same_votes / total_votes if total_votes else 0.0
    different_share = different_votes / total_votes if total_votes else 0.0
    st.subheader("Overview at a glance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Shared votes analysed", total_votes)
    col2.metric(
        "Votes with identical positions",
        same_votes,
        delta=f"{same_share:.0%}" if total_votes else None,
    )
    col3.metric(
        "Votes with differing positions",
        different_votes,
        delta=f"{different_share:.0%}" if total_votes else None,
    )
    st.caption(
        "Metrics summarise all shared votes that match the sidebar filters. "
        "Use the options below to focus on specific agreement patterns."
    )
    st.markdown("### Focus the analysis")
    st.caption(
        "Choose which voting pattern to emphasise throughout the charts and tables."
    )

    filter_index = filter_options.index(filter_option)
    filter_option = st.radio(
        "Filter votes by agreement",
        filter_options,
        index=filter_index,
        horizontal=True,
        help="Limit the analysis to votes where the selected entities agreed or differed.",
    )

    filtered = vote_matrix.copy()
    filtered.attrs = vote_matrix.attrs
    if filter_option == "Only same":
        filtered = filtered[filtered["agreement"] == "Same"]
    elif filter_option == "Only different":
        filtered = filtered[filtered["agreement"] == "Different"]

    breakdown = category_breakdown(filtered)
    summary_metrics = build_summary_metrics(total_votes, same_votes, different_votes)
    pairwise_df = pairwise_agreement(filtered, selected_columns)
    heatmap_chart = pairwise_agreement_heatmap(filtered, selected_columns)

    if not pairwise_df.empty:
        evaluation_df = pairwise_df.copy()
        evaluation_df["_total"] = evaluation_df["Same"] + evaluation_df["Different"]
        evaluation_df = evaluation_df[evaluation_df["_total"] > 0]
        if not evaluation_df.empty:
            evaluation_df["_share"] = evaluation_df.apply(
                lambda row: row["Same"] / row["_total"], axis=1
            )
            best_pair = evaluation_df.sort_values("_share", ascending=False).iloc[0]
            worst_pair = evaluation_df.sort_values("_share", ascending=True).iloc[0]
            st.caption(
                f"Highest alignment: {best_pair['Pair']} ({best_pair['Agreement rate']}). "
                f"Lowest alignment: {worst_pair['Pair']} ({worst_pair['Agreement rate']})."
            )

    st.markdown("### Explore the results")
    st.caption(
        "Switch between the tabs to compare subjects, pairwise agreement, and the full vote log."
    )

    pair_label_to_members: Dict[str, Tuple[str, str]] = {}
    pair_label_to_token: Dict[str, str] = {}
    pair_options = ["No pair selected"]
    for left, right in combinations(selected_columns, 2):
        label = f"{left} ↔ {right}"
        token = encode_pair_token(left, right)
        pair_label_to_members[label] = (left, right)
        pair_label_to_token[label] = token
        pair_options.append(label)

    pair_default_label = "No pair selected"
    pair_param = params.get("pair", "")
    if pair_param:
        decoded = decode_pair_token(pair_param)
        if decoded:
            candidate_label = f"{decoded[0]} ↔ {decoded[1]}"
            if candidate_label in pair_label_to_members:
                pair_default_label = candidate_label

    detail_focus_label: str | None = None
    selected_pair_label: str | None = None
    selected_pair_token: str | None = None

    summary_tab, agreement_tab, votes_tab = st.tabs(
        ["📚 Subjects", "🤝 Agreement", "🗳️ Votes"]
    )

    with summary_tab:
        st.markdown("#### Subject mix overview")
        st.caption(
            "Spot the policy areas that drive agreement or disagreement across the selected entities."
        )
        if breakdown.empty:
            st.info(
                "Subject metadata is unavailable for the current selection of shared votes."
            )
        else:
            category_chart = category_agreement_chart(breakdown)
            if category_chart is not None:
                _render_full_width_altair(category_chart)

            subjects_in_breakdown = breakdown["Top-level subject"].tolist()
            category_options = ["All categories", *subjects_in_breakdown]
            default_focus_option = (
                focus_default
                if focus_default in subjects_in_breakdown
                else "All categories"
            )
            focus_selection = st.selectbox(
                "Drill down on a subject",
                options=category_options,
                index=category_options.index(default_focus_option),
                help="Restrict the detailed vote table to a single top-level OEIL subject.",
            )
            if focus_selection != "All categories":
                detail_focus_label = focus_selection

            st.dataframe(
                breakdown,
                width="stretch",
                hide_index=True,
            )

    with agreement_tab:
        st.markdown("#### Pair alignment insights")
        st.caption(
            "See how the selected entities line up with one another and track agreement trends over time."
        )
        if heatmap_chart is not None:
            _render_full_width_altair(heatmap_chart)
            st.caption(
                "Darker cells indicate a higher share of matching votes between the pair."
            )

        if pair_label_to_members:
            pair_index = pair_options.index(pair_default_label)
            selected_pair_label = st.selectbox(
                "Highlight an entity pair",
                options=pair_options,
                index=pair_index,
                help="Pick a pair to spotlight their agreement record and time-series trend.",
            )
            if selected_pair_label and selected_pair_label != "No pair selected":
                selected_pair_token = pair_label_to_token.get(selected_pair_label)
                pair_members = pair_label_to_members.get(selected_pair_label)
                if pair_members:
                    left_name, right_name = pair_members
                trend_df = pairwise_trend(
                    filtered, left_name, right_name, frequency=trend_frequency
                )
                trend_chart = pairwise_trend_chart(trend_df)
                if trend_chart is not None:
                    _render_full_width_altair(trend_chart)
                elif trend_df.empty:
                    st.info(
                        "Not enough timestamped votes remain to draw a trend for this pair."
                    )
        else:
            st.info("At least two entities are required to analyse pairwise alignment.")

        if not pairwise_df.empty:
            st.dataframe(
                pairwise_df,
                width="stretch",
                hide_index=True,
            )

    subject_filtered_votes = filtered.copy()
    if detail_focus_label:
        subject_filtered_votes = subject_filtered_votes[
            subject_filtered_votes["top_subjects"].apply(
                lambda cats: detail_focus_label in _to_category_list(cats)
            )
        ]

    if detail_focus_label and subject_filtered_votes.empty:
        st.info(
            "No votes remain after applying the current subject drill-down. "
            "Clear the drill-down selection to view all filtered votes."
        )

    with votes_tab:
        st.markdown("#### Vote-by-vote detail")
        st.caption(
            "Inspect every shared vote side by side. Use the download buttons to keep a copy of the current view."
        )
        detail_caption = (
            "Positions use the official roll-call codes: FOR, AGAINST, ABSTENTION, "
            "and DID_NOT_VOTE."
        )
        if detail_focus_label:
            detail_caption += f" Showing only votes tagged with '{detail_focus_label}'."
        st.caption(detail_caption)

        vote_title_query = st.text_input(
            "Search vote titles",
            value=vote_title_query_default,
            placeholder="e.g. Ukraine",
            help=(
                "Search across vote titles, procedure names, subjects, and geography tags. "
                "Use uppercase OR to keep alternatives (e.g. 'Ukraine OR Russia')."
            ),
        )

        filtered_for_display = subject_filtered_votes.copy()
        query_groups = parse_search_query(vote_title_query)
        search_columns = [
            column
            for column in [
                "display_title",
                "procedure_title",
                "procedure_reference",
                "subjects_display",
                "top_subjects_display",
                "geo_areas_display",
                "eurovoc_concepts_display",
                "description",
            ]
            if column in subject_filtered_votes.columns
        ]

        if query_groups and search_columns:
            mask = build_search_mask(
                subject_filtered_votes, query_groups, search_columns
            )
            filtered_for_display = subject_filtered_votes[mask].copy()
        elif not query_groups:
            filtered_for_display = subject_filtered_votes.copy()

        sync_query_params(
            detail_focus_label,
            selected_pair_token,
            vote_query_value=vote_title_query.strip() or None,
        )

        if (
            query_groups
            and filtered_for_display.empty
            and not subject_filtered_votes.empty
        ):
            st.info(
                "No votes match the current title search. Try different keywords or clear the search field."
            )

        display_df = format_vote_table(filtered_for_display, selected_columns)
        styled_or_plain = style_vote_table(display_df)
        st.dataframe(
            styled_or_plain,
            width="stretch",
            hide_index=True,
        )
        if isinstance(styled_or_plain, pd.DataFrame) and styled_or_plain.attrs.get(
            "styling_skipped"
        ):
            st.caption(
                "Styling disabled for very large tables. Download the CSV for full details."
            )

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download table as CSV",
            data=csv_data,
            file_name="eu_parliament_vote_comparison.csv",
            mime="text/csv",
        )

        export_blob = build_export_package(
            display_df, summary_metrics, breakdown, pairwise_df
        )
        st.download_button(
            "Download comparison package (.zip)",
            data=export_blob,
            file_name="eu_parliament_vote_comparison.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
