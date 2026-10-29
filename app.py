from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Homicide Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
# Use the full CSV if present, otherwise the compressed copy stored in the repo.
DATA_PATH = next((p for p in (APP_DIR / "database.csv", APP_DIR / "database.csv.gz") if p.exists()), APP_DIR / "database.csv.gz")

# Consistent styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25);
        padding: 14px 16px; border-radius: 12px;}
    div[data-testid="stMetricLabel"] {font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading and preparing the CSV…")
def load_data(file_path: str, modified_time: float) -> pd.DataFrame:
    """Read once and cache the data. The mtime argument refreshes cache if the CSV changes."""
    data = pd.read_csv(file_path, low_memory=False)
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )

    # Convert numeric columns without forcing invalid values to zero.
    for col in ["year", "victim_age", "perpetrator_age", "victim_count", "perpetrator_count"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Match the notebook's handling of special age codes.
    if "victim_age" in data.columns:
        data.loc[data["victim_age"] >= 998, "victim_age"] = np.nan
    if "perpetrator_age" in data.columns:
        data.loc[data["perpetrator_age"] < 8, "perpetrator_age"] = np.nan
        data.loc[data["perpetrator_age"] > 100, "perpetrator_age"] = np.nan

    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype("string").str.strip()

    if "year" in data.columns:
        data["year"] = data["year"].astype("Int64")

    if "month" in data.columns:
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        month_map = {month: i for i, month in enumerate(month_order, start=1)}
        data["month_num"] = data["month"].map(month_map).astype("Int64")

    return data


def safe_count(series: pd.Series, value: str) -> int:
    return int(series.astype("string").str.casefold().eq(value.casefold()).sum())


def make_empty_chart(message: str = "No data available for these filters."):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=25, b=20))
    return fig


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=35),
        legend_title_text="",
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(120,130,140,0.18)")
    return fig


st.title("Homicide Data Explorer")
st.caption("Explore homicide records from 1980–2014. Counts represent records in this dataset, not necessarily unique incidents or population-adjusted crime rates.")

if not DATA_PATH.exists():
    st.error("I couldn't find `database.csv` or `database.csv.gz` next to `app.py`.")
    st.markdown(
        "Place **app.py** and **database.csv.gz** (or database.csv) in the same folder, then run `streamlit run app.py`."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Couldn't read the CSV: {exc}")
    st.stop()

required = {"year", "state", "crime_solved", "weapon", "perpetrator_sex", "perpetrator_age", "victim_age", "relationship", "month"}
missing = sorted(required - set(df.columns))
if missing:
    st.error("Some expected columns are missing after column-name cleanup: " + ", ".join(missing))
    st.write("Columns found:", list(df.columns))
    st.stop()

# The source dataset is expected to cover 1980–2014. Keep valid years only for the dashboard.
base = df[df["year"].between(1980, 2014)].copy()
if base.empty:
    st.warning("No records with years from 1980 to 2014 were found.")
    st.stop()

# Sidebar controls
PAGES = ["📈 Overview", "👥 People & Weapons", "🗺️ States & Time", "📄 Data Table"]
st.sidebar.title("Navigate")
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
st.sidebar.divider()

st.sidebar.subheader("Filters")
year_min = int(base["year"].min())
year_max = int(base["year"].max())
year_range = st.sidebar.slider(
    "Years covered", min_value=year_min, max_value=year_max,
    value=(year_min, year_max), step=1,
)
YEARS = f"{year_range[0]}–{year_range[1]}"

all_states = sorted(base["state"].dropna().astype(str).unique().tolist())
if "selected_states" not in st.session_state:
    st.session_state["selected_states"] = all_states


def _select_all_states():
    st.session_state["selected_states"] = all_states


def _clear_states():
    st.session_state["selected_states"] = []


n_sel = len(st.session_state["selected_states"])
with st.sidebar.expander(f"States ({n_sel} of {len(all_states)} selected)"):
    b1, b2 = st.columns(2)
    b1.button("Select all", on_click=_select_all_states, use_container_width=True)
    b2.button("Clear", on_click=_clear_states, use_container_width=True)
    selected_states = st.multiselect(
        "Choose states", options=all_states, key="selected_states", label_visibility="collapsed",
        placeholder="Pick one or more states",
    )
status_options = ["All", "Yes", "No"]
selected_status = st.sidebar.selectbox(
    "Show cases", options=status_options, index=0,
    format_func={"All": "All cases", "Yes": "Solved only", "No": "Unsolved only"}.get,
)
st.sidebar.caption("Age charts use valid recorded ages only. Many unsolved records have unknown perpetrator ages.")

filtered = base[base["year"].between(year_range[0], year_range[1])].copy()
if selected_states:
    filtered = filtered[filtered["state"].isin(selected_states)]
else:
    filtered = filtered.iloc[0:0]
if selected_status != "All":
    filtered = filtered[filtered["crime_solved"].astype("string").str.casefold().eq(selected_status.casefold())]

if filtered.empty:
    st.warning("No records match the current filters. Select at least one state or widen the year/status filters.")
    st.stop()

st.sidebar.divider()
st.sidebar.caption(f"Showing {len(filtered):,} of {len(base):,} records (1980–2014).")

# Derived summaries shared by pages
solved_mask = filtered["crime_solved"].astype("string").str.casefold().eq("yes")
unsolved_mask = filtered["crime_solved"].astype("string").str.casefold().eq("no")
known_status = filtered["crime_solved"].astype("string").str.casefold().isin(["yes", "no"])
solved_rate = (solved_mask.sum() / known_status.sum() * 100) if known_status.sum() else 0.0
weapon_clean = filtered["weapon"].astype("string").str.strip()
weapon_clean = weapon_clean[weapon_clean.notna() & weapon_clean.ne("") & weapon_clean.str.casefold().ne("unknown")]
most_common_weapon = weapon_clean.value_counts().index[0] if not weapon_clean.empty else "N/A"

if page == PAGES[0]:
    st.subheader(f"The big picture, {YEARS}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total homicide records", f"{len(filtered):,}")
    c2.metric("Share of cases solved", f"{solved_rate:.1f}%", help="Solved records divided by records that are marked solved or unsolved.")
    c3.metric("Unsolved cases", f"{int(unsolved_mask.sum()):,}")
    c4.metric("Most common weapon", str(most_common_weapon), help="Excludes records where the weapon is unknown.")

    left, right = st.columns([1.5, 1])
    with left:
        yearly = filtered.groupby("year").size().reindex(range(year_range[0], year_range[1] + 1), fill_value=0).rename_axis("Year").reset_index(name="Records")
        fig = px.line(yearly, x="Year", y="Records", markers=True, title="How homicide records changed year by year")
        fig.update_yaxes(title="Number of homicide records")
        fig.update_traces(line_width=3, marker_size=5)
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)
    with right:
        status_counts = filtered["crime_solved"].astype("string").value_counts(dropna=False).rename_axis("Case status").reset_index(name="Records")
        status_counts = status_counts[status_counts["Case status"].isin(["Yes", "No"])]
        status_counts["Case status"] = status_counts["Case status"].map({"Yes": "Solved", "No": "Unsolved"})
        fig = px.pie(status_counts, names="Case status", values="Records", hole=0.55, title="Solved vs Unsolved Cases")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)

    left, right = st.columns(2)
    with left:
        top_states = filtered["state"].value_counts().head(10).sort_values().rename_axis("State").reset_index(name="Records")
        fig = px.bar(top_states, x="Records", y="State", orientation="h", title="10 states with the most homicide records", text="Records")
        fig.update_xaxes(title="Number of homicide records")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    with right:
        if "month_num" in filtered.columns:
            month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            month_data = filtered.dropna(subset=["year", "month_num"]).groupby(["year", "month_num"]).size().rename("Records").reset_index()
            pivot = month_data.pivot(index="year", columns="month_num", values="Records").reindex(index=range(year_range[0], year_range[1] + 1), columns=range(1, 13), fill_value=0).fillna(0)
            fig = px.imshow(pivot, x=month_order, y=pivot.index, aspect="auto", color_continuous_scale="RdBu_r", labels=dict(x="Month", y="Year", color="Records"), title="Which months and years had the most homicides (darker red = more)")
            fig.update_layout(height=430, margin=dict(l=20, r=20, t=55, b=45))
            st.plotly_chart(fig, use_container_width=True)

elif page == PAGES[1]:
    st.subheader("Who is involved and what weapons are used")
    st.caption("Age charts use records with a known, valid age only (perpetrator ages below 8 or above 100 are excluded).")

    left, right = st.columns(2)
    with left:
        ages = filtered["perpetrator_age"].dropna()
        ages = ages[ages.between(8, 100)]
        if not ages.empty:
            age_bins = list(range(5, 101, 5)) + [105]
            age_labels = [f"{i}-{i+4}" for i in range(5, 100, 5)] + ["100-104"]
            grouped = pd.cut(ages, bins=age_bins, labels=age_labels, right=False, include_lowest=True)
            counts = grouped.value_counts(sort=False).reindex(age_labels, fill_value=0).rename_axis("Age group").reset_index(name="Records")
            peak_group = counts.loc[counts["Records"].idxmax(), "Age group"]
            fig = px.bar(counts, x="Age group", y="Records", title=f"Perpetrators are most often aged {peak_group}")
            fig.update_xaxes(tickangle=-45, title="Perpetrator age group (years)")
            fig.update_yaxes(title="Number of records")
            st.plotly_chart(style_fig(fig, 410), use_container_width=True)
        else:
            st.plotly_chart(make_empty_chart("No valid perpetrator ages in this selection."), use_container_width=True)
    with right:
        gap_data = filtered[["perpetrator_age", "victim_age"]].dropna().copy()
        gap_data = gap_data[gap_data["perpetrator_age"].between(8, 100) & gap_data["victim_age"].between(1, 100)]
        if not gap_data.empty:
            gap_data["Age gap"] = (gap_data["perpetrator_age"] - gap_data["victim_age"]).abs()
            gap_bins = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 100]
            gap_labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40+"]
            gap_data["Age gap group"] = pd.cut(gap_data["Age gap"], bins=gap_bins, labels=gap_labels)
            pct = gap_data["Age gap group"].value_counts(normalize=True, sort=False).reindex(gap_labels, fill_value=0).mul(100).rename_axis("Age gap").reset_index(name="Percent")
            top_gap = pct.loc[pct["Percent"].idxmax()]
            fig = px.bar(pct, x="Age gap", y="Percent", title=f"{top_gap['Percent']:.0f}% of cases: perpetrator and victim are {top_gap['Age gap']} years apart in age", text="Percent")
            fig.update_xaxes(title="Age difference between perpetrator and victim (years)")
            fig.update_yaxes(title="Share of records (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            st.plotly_chart(style_fig(fig, 410), use_container_width=True)
        else:
            st.plotly_chart(make_empty_chart("Not enough valid victim/perpetrator ages for this selection."), use_container_width=True)

    st.markdown("#### Which weapons do male and female perpetrators use?")
    wdata = filtered[["perpetrator_sex", "weapon"]].dropna().copy()
    wdata["perpetrator_sex"] = wdata["perpetrator_sex"].astype("string").str.strip()
    wdata["weapon"] = wdata["weapon"].astype("string").str.strip()
    wdata = wdata[wdata["perpetrator_sex"].isin(["Male", "Female"]) & wdata["weapon"].ne("") & wdata["weapon"].str.casefold().ne("unknown")]
    top_weapons = wdata["weapon"].value_counts().head(8).index.tolist()
    wplot = wdata[wdata["weapon"].isin(top_weapons)]
    if not wplot.empty:
        weapon_counts = wplot.groupby(["weapon", "perpetrator_sex"]).size().rename("Records").reset_index()
        fig = px.bar(weapon_counts, x="weapon", y="Records", color="perpetrator_sex", barmode="group", category_orders={"weapon": top_weapons, "perpetrator_sex": ["Male", "Female"]}, title="The 8 most common weapons, split by perpetrator sex", labels={"perpetrator_sex": "Perpetrator sex"})
        fig.update_xaxes(tickangle=-30, title="Weapon")
        fig.update_yaxes(title="Number of records")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    else:
        st.plotly_chart(make_empty_chart("No male/female weapon records in this selection."), use_container_width=True)
    st.info("Counts only include records with a known weapon and a male or female perpetrator. Male records far outnumber female ones, so compare the mix of weapons, not the raw totals.")

elif page == PAGES[2]:
    st.subheader("Where and when homicides happen, and where cases get solved")
    left, right = st.columns(2)
    with left:
        state_counts = filtered["state"].value_counts().head(10).sort_values().rename_axis("State").reset_index(name="Records")
        fig = px.bar(state_counts, x="Records", y="State", orientation="h", title="10 states with the most homicide records", text="Records")
        fig.update_xaxes(title="Number of homicide records")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    with right:
        state_data = filtered[filtered["crime_solved"].astype("string").isin(["Yes", "No"])].copy()
        state_stats = state_data.groupby("state").agg(Total_Cases=("crime_solved", "size"), Solved_Cases=("crime_solved", lambda x: int((x == "Yes").sum()))).reset_index()
        state_stats["Solved Rate (%)"] = np.where(state_stats["Total_Cases"] > 0, state_stats["Solved_Cases"] / state_stats["Total_Cases"] * 100, np.nan)
        state_stats = state_stats[state_stats["Total_Cases"] >= 100]
        if not state_stats.empty:
            rank = st.radio("Ranking", ["Best at solving cases", "Worst at solving cases"], horizontal=True, label_visibility="collapsed")
            if rank.startswith("Best"):
                best = state_stats.nlargest(10, "Solved Rate (%)").sort_values("Solved Rate (%)")
            else:
                best = state_stats.nsmallest(10, "Solved Rate (%)").sort_values("Solved Rate (%)", ascending=False)
            fig = px.bar(
                best, x="Solved Rate (%)", y="state", orientation="h",
                title=f"10 states {'with the highest' if rank.startswith('Best') else 'with the lowest'} share of cases solved",
                hover_data=["Total_Cases", "Solved_Cases"], labels={"state": "State", "Total_Cases": "Total cases", "Solved_Cases": "Solved cases"},
            )
            fig.update_xaxes(range=[0, 100], title="Cases solved (%)  ·  states with 100+ cases")
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
        else:
            st.plotly_chart(make_empty_chart("No states meet the 100-record minimum."), use_container_width=True)

    st.markdown("#### Which months and years were worst in one state?")
    st.caption("Counts are scaled 0–10 within the chosen state (10 = its busiest month), so states of different size can be compared by pattern.")
    state_pick = st.selectbox("Choose a state",sorted(filtered["state"].dropna().unique()), index=0)
    sm = filtered[(filtered["state"] == state_pick)].dropna(subset=["year", "month_num"])
    sm_pivot = (
        sm.groupby(["year", "month_num"]).size().unstack(fill_value=0)
        .reindex(index=range(year_range[0], year_range[1] + 1), columns=range(1, 13), fill_value=0)
    )
    if sm_pivot.to_numpy().max() > 0:
        sm_pivot = sm_pivot / sm_pivot.to_numpy().max() * 10
        fig = px.imshow(
            sm_pivot, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=sm_pivot.index, aspect="auto", zmin=0, zmax=10, color_continuous_scale="RdBu_r",
            labels=dict(x="Month", y="Year", color="Scaled (0–10)"), title=f"{state_pick}: busiest months and years (10 = peak)",
        )
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=55, b=35))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(make_empty_chart(), use_container_width=True)

    st.markdown("#### Does the weapon affect whether a case gets solved?")
    weapon_stats = filtered.dropna(subset=["weapon", "crime_solved"]).copy()
    weapon_stats["weapon"] = weapon_stats["weapon"].astype("string").str.strip()
    weapon_stats = weapon_stats[weapon_stats["weapon"].ne("") & weapon_stats["weapon"].str.casefold().ne("unknown") & weapon_stats["crime_solved"].astype("string").isin(["Yes", "No"])]
    weapon_summary = weapon_stats.groupby("weapon").agg(
        Total_Cases=("crime_solved", "size"),
        Solved_Cases=("crime_solved", lambda x: int((x == "Yes").sum())),
    ).reset_index()
    weapon_summary = weapon_summary[weapon_summary["Total_Cases"] >= 500].copy()
    if not weapon_summary.empty:
        weapon_summary["Solved Rate (%)"] = weapon_summary["Solved_Cases"] / weapon_summary["Total_Cases"] * 100
        top_weapon_rates = weapon_summary.nlargest(10, "Solved Rate (%)").sort_values("Solved Rate (%)")
        fig = px.bar(top_weapon_rates, x="Solved Rate (%)", y="weapon", orientation="h", title="Weapons with the highest share of cases solved", hover_data=["Total_Cases", "Solved_Cases"], labels={"weapon": "Weapon", "Total_Cases": "Total cases", "Solved_Cases": "Solved cases"})
        fig.update_xaxes(range=[0, 100], title="Cases solved (%)  ·  weapons with 500+ cases")
        st.plotly_chart(style_fig(fig, 450), use_container_width=True)
    else:
        st.plotly_chart(make_empty_chart("No weapon types meet the 500-record minimum."), use_container_width=True)

    st.markdown("#### Where do the most unsolved cases pile up?")
    unsolved_by_state = filtered[filtered["crime_solved"].astype("string").eq("No")]["state"].value_counts().head(15).sort_values().rename_axis("State").reset_index(name="Unsolved records")
    if not unsolved_by_state.empty:
        fig = px.bar(unsolved_by_state, x="Unsolved records", y="State", orientation="h", title="15 states with the most unsolved homicide cases")
        fig.update_xaxes(title="Number of unsolved cases")
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)
    else:
        st.plotly_chart(make_empty_chart("No unsolved records in this selection."), use_container_width=True)

    st.caption("'Cases solved (%)' = solved cases ÷ cases marked solved or unsolved. It says nothing about how quickly cases were solved.")

else:  # Data Table
    st.subheader("Browse the records behind the charts")
    st.write("The table follows the sidebar filters (first 500 rows shown; the download has all filtered rows). Impossible ages are blanked out; the original CSV is not modified.")
    st.dataframe(filtered.head(500), use_container_width=True, height=500)
    st.download_button(
        "Download filtered records (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_homicide_records.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Source: Homicide Reports dataset. Interpret record counts carefully: this dataset does not provide population denominators, and a record is not guaranteed to represent a unique incident.")
