"""Streamlit application to compare European Parliament member votes."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

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
    return pd.read_csv(DATA_DIR / "member_votes.csv.gz")


@lru_cache(maxsize=1)
def member_lookup() -> Dict[int, str]:
    members = load_members()
    return dict(zip(members["id"], members["full_name"]))


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
    return combined


def format_vote_table(df: pd.DataFrame, selected_member_ids: List[int]) -> pd.DataFrame:
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
            "agreement",
            "shared_position",
            *selected_names,
        ]
    ].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df = display_df.rename(
        columns={
            "display_title": "Vote title",
            "procedure_reference": "Procedure ref",
            "procedure_title": "Procedure title",
            "is_main": "Main vote",
            "shared_position": "Shared position",
        }
    )
    return display_df


def main() -> None:
    st.set_page_config(page_title="EU Parliament Vote Comparator", layout="wide")
    st.title("European Parliament Vote Comparator")

    members = load_members()
    member_names = members["full_name"].tolist()

    st.sidebar.header("Select members")
    selected = st.sidebar.multiselect(
        "Members of the European Parliament",
        options=member_names,
        max_selections=6,
        help="Choose up to six members to compare their voting records.",
    )

    name_to_id = dict(zip(members["full_name"], members["id"]))
    selected_ids = [name_to_id[name] for name in selected]

    if not selected_ids:
        st.info("Select at least one member to begin exploring voting patterns.")
        return

    vote_matrix = build_vote_matrix(selected_ids)

    if vote_matrix.empty:
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

    filter_option = st.radio(
        "Filter votes by agreement",
        ("All votes", "Only same", "Only different"),
        horizontal=True,
    )

    filtered = vote_matrix.copy()
    if filter_option == "Only same":
        filtered = filtered[filtered["agreement"] == "Same"]
    elif filter_option == "Only different":
        filtered = filtered[filtered["agreement"] == "Different"]

    display_df = format_vote_table(filtered, selected_ids)

    st.subheader("Detailed comparison")
    st.dataframe(
        display_df,
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
