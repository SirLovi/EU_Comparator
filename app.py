"""Streamlit application to compare European Parliament member votes."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import altair as alt

DATA_DIR = Path(__file__).resolve().parent / "DATA"


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
        "is_main",
    ]
    return votes[columns]


@st.cache_data(show_spinner=False)
def load_member_votes() -> pd.DataFrame:
    """Load the vote positions of all members."""
    return pd.read_csv(
        DATA_DIR / "member_votes.csv.gz",
        usecols=["vote_id", "member_id", "position"],
        dtype={"vote_id": "int64", "member_id": "int64", "position": "category"},
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


@lru_cache(maxsize=1)
def member_lookup() -> Dict[int, str]:
    members = load_members()
    return dict(zip(members["id"], members["full_name"]))


def _to_category_list(value: object) -> List[str]:
    """Normalise stored category tuples to a list of readable labels."""
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def build_vote_matrix(selected_member_ids: List[int]) -> pd.DataFrame:
    """Create a wide table of common votes for the selected members."""
    if not selected_member_ids:
        return pd.DataFrame()

    member_votes = load_member_votes()
    subset = member_votes[member_votes["member_id"].isin(selected_member_ids)].copy()

    # Keep only votes where every selected member cast a position.
    participation = subset.groupby("vote_id")["member_id"].nunique()
    complete_vote_ids = participation[participation == len(selected_member_ids)].index
    if complete_vote_ids.empty:
        return pd.DataFrame()

    subset = subset[subset["vote_id"].isin(complete_vote_ids)]

    members = load_members().set_index("id")
    subset = subset.merge(
        members[["full_name"]], left_on="member_id", right_index=True, how="left"
    )

    pivot = subset.pivot_table(
        index="vote_id", columns="full_name", values="position", aggfunc="first"
    )

    # Preserve the order in which members were selected in the UI.
    selected_names = [member_lookup()[mid] for mid in selected_member_ids]
    pivot = pivot.reindex(columns=selected_names)

    vote_details = load_votes().set_index("id")
    pivot = pivot.join(vote_details, how="left")

    def summarise(row: pd.Series) -> pd.Series:
        member_positions = row[selected_names].dropna()
        unique_positions = set(member_positions.tolist())
        all_same = len(unique_positions) == 1
        shared_position = member_positions.iloc[0] if all_same else None
        return pd.Series(
            {
                "agreement": "Same" if all_same else "Different",
                "shared_position": shared_position,
            }
        )

    summary = pivot.apply(summarise, axis=1)
    combined = pd.concat([pivot, summary], axis=1)
    combined = combined.reset_index().rename(columns={"index": "vote_id"})
    combined = combined.sort_values("timestamp", ascending=False)

    categories = load_vote_subjects()
    combined = combined.merge(categories, on="vote_id", how="left")

    def ensure_tuple(value: object) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return tuple()

    for column in ("subjects", "top_subjects"):
        combined[column] = combined[column].apply(ensure_tuple)

    for column in ("subjects_display", "top_subjects_display"):
        combined[column] = combined[column].fillna("—")

    combined = combined.sort_values("timestamp", ascending=False)
    return combined


def format_vote_table(df: pd.DataFrame, selected_member_ids: List[int]) -> pd.DataFrame:
    """Select, rename, and prettify columns for display & download."""
    if df.empty:
        return df

    selected_names = [member_lookup()[mid] for mid in selected_member_ids]
    display_df = df[
        [
            "timestamp",
            "display_title",
            "procedure_reference",
            "procedure_title",
            "is_main",
            "top_subjects_display",
            "subjects_display",
            "agreement",
            "shared_position",
            *selected_names,
        ]
    ].copy()

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

    return display_df


def style_vote_table(
    display_df: pd.DataFrame,
) -> pd.io.formats.style.Styler | pd.DataFrame:
    """Apply colour styling to highlight agreement and vote positions.
    Returns a Styler if styling is applicable; otherwise the raw DataFrame.
    """
    if display_df.empty:
        return display_df

    position_styles: dict[str | None, str] = {
        "FOR": "background-color: #d1e7dd; color: #0f5132; font-weight: 600;",
        "AGAINST": "background-color: #f8d7da; color: #842029; font-weight: 600;",
        "ABSTENTION": "background-color: #fff3cd; color: #664d03; font-weight: 600;",
        "DID_NOT_VOTE": "background-color: #e2e3e5; color: #41464b;",
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
        "Agreement",
        "Shared position",
    }
    position_columns = [col for col in display_df.columns if col not in base_columns]

    styler = display_df.style
    styler = styler.map(agreement_style, subset=["Agreement"])
    styler = styler.map(position_style, subset=["Shared position", *position_columns])
    styler = styler.set_properties(**{"white-space": "nowrap"})  # type: ignore[arg-type]
    return styler


def pairwise_agreement(
    df: pd.DataFrame, selected_member_ids: List[int]
) -> pd.DataFrame:
    """Return a summary of how often each pair of members agreed."""
    if df.empty or len(selected_member_ids) < 2:
        return pd.DataFrame()

    selected_names = [member_lookup()[mid] for mid in selected_member_ids]
    rows = []
    total_votes = len(df)

    for left, right in combinations(selected_names, 2):
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
    df: pd.DataFrame, selected_member_ids: List[int]
) -> alt.Chart | None:
    """Return a heatmap chart of pairwise agreement rates."""
    if df.empty or len(selected_member_ids) < 2:
        return None

    selected_names = [member_lookup()[mid] for mid in selected_member_ids]
    matrix = pd.DataFrame(
        1.0, index=selected_names, columns=selected_names, dtype=float
    )

    for left, right in combinations(selected_names, 2):
        subset = df[[left, right]].dropna()
        total = len(subset)
        if total:
            same = (subset[left] == subset[right]).sum()
            rate = same / total
        else:
            rate = float("nan")
        matrix.loc[left, right] = matrix.loc[right, left] = rate

    matrix = matrix.fillna(0.0)
    for name in selected_names:
        matrix.loc[name, name] = 1.0

    heatmap_data = matrix.reset_index().melt(
        id_vars="index", var_name="Member B", value_name="Agreement"
    )
    heatmap_data = heatmap_data.rename(columns={"index": "Member A"})

    chart = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X("Member B:N", sort=selected_names, title=""),
            y=alt.Y("Member A:N", sort=selected_names, title=""),
            color=alt.Color(
                "Agreement:Q",
                scale=alt.Scale(domain=[0, 1], scheme="blues"),
                legend=alt.Legend(title="Agreement"),
            ),
            tooltip=[
                alt.Tooltip("Member A", title="Member A"),
                alt.Tooltip("Member B", title="Member B"),
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


@st.cache_data(show_spinner=False)
def load_last_updated() -> str:
    """Get the timestamp of the source dataset if available."""
    path = DATA_DIR / "last_updated.txt"
    if not path.exists():
        return "unknown"
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    st.set_page_config(page_title="EU Parliament Vote Comparator", layout="wide")
    st.title("European Parliament Vote Comparator")

    members = load_members()
    member_names = members["full_name"].tolist()
    subjects_catalogue = load_vote_subjects()

    top_subject_series = subjects_catalogue.get("top_subjects")
    top_category_options = (
        sorted(
            {cat for entry in top_subject_series for cat in _to_category_list(entry)}
        )
        if top_subject_series is not None
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

    default_selected_members = [
        name for name in param_list("members") if name in member_names
    ]
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
    focus_default = params.get("focus", "")

    st.sidebar.header("Select members")
    selected = st.sidebar.multiselect(
        "Members of the European Parliament",
        options=member_names,
        max_selections=6,
        default=default_selected_members,
        help="Choose up to six members to compare their voting records.",
    )

    st.sidebar.divider()
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
        help=(
            "Narrow the subject list. Clear the search to show the full set of "
            "OEIL top-level subjects."
        ),
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
            f"{len(filtered_category_options)} matching of {len(top_category_options)} available subjects."
        )
    else:
        st.sidebar.caption(
            "Subject metadata is unavailable in the current dataset release."
        )

    st.sidebar.caption(f"Dataset last updated: {load_last_updated()}")

    name_to_id = dict(zip(members["full_name"], members["id"]))
    selected_ids = [name_to_id[name] for name in selected]

    def sync_query_params(focus_value: str | None = None) -> None:
        payload = {
            "main_only": [str(main_only).lower()],
            "filter": [filter_option],
        }
        if selected:
            payload["members"] = selected
        if category_search:
            payload["subject_query"] = [category_search]
        if selected_top_categories:
            payload["subjects"] = selected_top_categories
        if focus_value:
            payload["focus"] = [focus_value]

        candidate = {}
        for key, value in payload.items():
            if value:
                candidate[key] = value if isinstance(value, list) else [value]

        current = {key: params.get_all(key) for key in list(params.keys())}

        if current != candidate:
            params.clear()
            for key, value in payload.items():
                if not value:
                    continue
                if isinstance(value, list):
                    params[key] = value if len(value) != 1 else value[0]
                else:
                    params[key] = value

    if not selected_ids:
        sync_query_params()
        st.info("Select at least one member to begin exploring voting patterns.")
        return

    vote_matrix = build_vote_matrix(selected_ids)

    if main_only:
        vote_matrix = vote_matrix[vote_matrix["is_main"]]

    if selected_top_categories and not vote_matrix.empty:
        selected_set = set(selected_top_categories)
        vote_matrix = vote_matrix[
            vote_matrix["top_subjects"].apply(
                lambda cats: bool(set(_to_category_list(cats)) & selected_set)
            )
        ]

    if vote_matrix.empty:
        sync_query_params()
        st.warning(
            "No votes were found where all selected members participated. Try different members."
        )
        return

    same_votes = (vote_matrix["agreement"] == "Same").sum()
    different_votes = (vote_matrix["agreement"] == "Different").sum()

    total_votes = len(vote_matrix)
    st.subheader("At a glance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Shared votes analysed", total_votes)
    col2.metric("Votes with identical positions", same_votes)
    col3.metric("Votes with differing positions", different_votes)
    st.caption(
        "Metrics summarise all shared votes that match the sidebar filters. "
        "Use the options below to focus on specific agreement patterns."
    )

    filter_index = filter_options.index(filter_option)
    filter_option = st.radio(
        "Filter votes by agreement",
        filter_options,
        index=filter_index,
        horizontal=True,
    )

    filtered = vote_matrix.copy()
    if filter_option == "Only same":
        filtered = filtered[filtered["agreement"] == "Same"]
    elif filter_option == "Only different":
        filtered = filtered[filtered["agreement"] == "Different"]

    breakdown = category_breakdown(filtered)
    detail_focus: str | None = None
    if not breakdown.empty:
        st.subheader("Category breakdown")
        st.caption(
            "Counts reflect the top-level OEIL subjects represented in the current "
            "selection of shared votes."
        )
        category_chart = category_agreement_chart(breakdown)
        if category_chart is not None:
            st.altair_chart(category_chart, use_container_width=True)

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
            help="Highlight the detailed table with votes tagged to a specific subject.",
        )
        if focus_selection != "All categories":
            detail_focus = focus_selection

        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
        )

    heatmap_chart = pairwise_agreement_heatmap(filtered, selected_ids)
    if heatmap_chart is not None:
        st.subheader("Agreement heatmap")
        st.caption(
            "Colour intensity denotes how closely each pair aligns within the current selection."
        )
        st.altair_chart(heatmap_chart, use_container_width=True)

    pairwise = pairwise_agreement(filtered, selected_ids)
    if not pairwise.empty:
        st.subheader("Pairwise agreement overview")
        st.caption(
            "Agreement scores are based on the shared votes counted above "
            "and update automatically when filters change."
        )
        st.dataframe(
            pairwise,
            use_container_width=True,
            hide_index=True,
        )

    detail_filtered = filtered.copy()
    if detail_focus:
        detail_filtered = detail_filtered[
            detail_filtered["top_subjects"].apply(
                lambda cats: detail_focus in _to_category_list(cats)
            )
        ]

    sync_query_params(detail_focus)

    if detail_focus and detail_filtered.empty:
        st.info(
            "No votes remain after applying the current subject drill-down. "
            "Clear the drill-down selection to view all filtered votes."
        )

    display_df = format_vote_table(detail_filtered, selected_ids)

    st.subheader("Detailed comparison")
    detail_caption = (
        "Vote positions use the official roll-call codes: FOR, AGAINST, ABSTENTION, "
        "and DID_NOT_VOTE."
    )
    if detail_focus:
        detail_caption += f" Showing only votes tagged with '{detail_focus}'."
    st.caption(detail_caption)

    styled_or_plain = style_vote_table(display_df)
    st.dataframe(
        styled_or_plain,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        data=csv_data,
        file_name="eu_parliament_vote_comparison.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
