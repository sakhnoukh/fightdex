"""PokéCoach — Streamlit team recommendation UI."""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="PokéCoach",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
body { font-family: 'Segoe UI', sans-serif; }

.rec-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #243447 100%);
    border: 1px solid #3a4f6e;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.rec-card h4 { margin: 0 0 4px 0; color: #e8eaf0; font-size: 1.05rem; }
.rec-card .types { font-size: 0.75rem; margin-bottom: 6px; }
.type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.rank-label { font-size: 0.68rem; color: #5a7a9a; margin-bottom: 4px; }
.reason-list { margin: 6px 0 8px 0; padding: 0; list-style: none; }
.reason-list li { font-size: 0.82rem; color: #c8d8e8; margin: 3px 0; }
.reason-check { color: #27ae60; margin-right: 5px; font-weight: bold; }

.slot-empty { color: #55657a; font-style: italic; }

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a8dee;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #2a3a50;
}

.win-banner {
    background: linear-gradient(135deg, #1e2a3a 0%, #243447 100%);
    border: 1px solid #3a4f6e;
    border-radius: 12px;
    padding: 16px 24px;
    margin: 12px 0;
}
.win-formula {
    background: #1a2535;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: monospace;
    font-size: 0.82rem;
    color: #a8b8cc;
    margin: 8px 0;
    line-height: 1.6;
}

.matchup-table { border-collapse: collapse; width: 100%; }
.matchup-table th {
    background: #1a2535;
    color: #8899aa;
    font-size: 0.72rem;
    padding: 6px 10px;
    text-align: center;
    font-weight: 600;
}
.matchup-table td {
    text-align: center;
    padding: 6px 10px;
    font-weight: 700;
    font-size: 0.82rem;
}
.matchup-row-label {
    color: #c8d8e8;
    font-size: 0.72rem;
    text-align: right;
    padding-right: 10px;
    white-space: nowrap;
}

.onboarding-card {
    background: linear-gradient(135deg, #1a2535 0%, #1e2a3a 100%);
    border: 1px solid #2a3a50;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 8px 0;
}
.onboarding-card h3 { color: #e8eaf0; margin: 0 0 16px 0; font-size: 1.1rem; }
.onboarding-step { display: flex; align-items: flex-start; margin: 10px 0; }
.step-num {
    background: #5a8dee;
    color: #fff;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 10px;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text { font-size: 0.88rem; color: #a8b8cc; }

.stale-banner {
    background: #2d2a1a;
    border: 1px solid #8a7a20;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: #d4c44a;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Type colours ──────────────────────────────────────────────────────────────
TYPE_COLOURS: dict[str, tuple[str, str]] = {
    "fire":     ("#e74c3c", "#fff"),
    "water":    ("#3498db", "#fff"),
    "grass":    ("#27ae60", "#fff"),
    "electric": ("#f39c12", "#000"),
    "psychic":  ("#9b59b6", "#fff"),
    "ice":      ("#74b9ff", "#000"),
    "dragon":   ("#6c5ce7", "#fff"),
    "dark":     ("#2d3436", "#fff"),
    "fairy":    ("#fd79a8", "#000"),
    "fighting": ("#d35400", "#fff"),
    "poison":   ("#8e44ad", "#fff"),
    "ground":   ("#e67e22", "#fff"),
    "flying":   ("#74b9ff", "#000"),
    "bug":      ("#a3cb38", "#000"),
    "rock":     ("#95a5a6", "#000"),
    "ghost":    ("#636e72", "#fff"),
    "steel":    ("#778ca3", "#fff"),
    "normal":   ("#b2bec3", "#000"),
}

ALL_TYPES = sorted(TYPE_COLOURS.keys())

OPPONENT_PRESETS: dict[str, list[str]] = {
    "Rain":       ["Kyogre", "Pelipper", "Urshifu-Rapid-Strike", "Incineroar", "Flutter Mane", "Farigiraf"],
    "Cal-Shadow": ["Calyrex-Shadow", "Incineroar", "Rillaboom", "Flutter Mane", "Urshifu-Rapid-Strike", "Amoonguss"],
    "Miraidon":   ["Miraidon", "Flutter Mane", "Iron Hands", "Incineroar", "Tornadus", "Rillaboom"],
    "Cal-Ice TR": ["Calyrex-Ice", "Ursaluna-Bloodmoon", "Incineroar", "Amoonguss", "Farigiraf", "Rillaboom"],
    "Chi-Yu":     ["Chi-Yu", "Flutter Mane", "Incineroar", "Urshifu-Rapid-Strike", "Rillaboom", "Amoonguss"],
    "Chien-Pao":  ["Chien-Pao", "Flutter Mane", "Incineroar", "Urshifu-Rapid-Strike", "Rillaboom", "Tornadus"],
}

CELL_BG: list[tuple[float, str, str]] = [
    (4.0, "#27ae60", "#fff"),
    (2.0, "#6ab04c", "#fff"),
    (1.0, "#778ca3", "#fff"),
    (0.5, "#e67e22", "#fff"),
    (0.25, "#e74c3c", "#fff"),
]

MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "Balanced":      {"synergy": 0.45, "counter": 0.30, "viability": 0.25, "content": 0.15},
    "Counter-heavy": {"synergy": 0.15, "counter": 0.60, "viability": 0.15, "content": 0.10},
    "Meta-focused":  {"synergy": 0.20, "counter": 0.10, "viability": 0.60, "content": 0.10},
    "Synergy-first": {"synergy": 0.65, "counter": 0.15, "viability": 0.15, "content": 0.05},
}


def type_badge(t: str) -> str:
    bg, fg = TYPE_COLOURS.get(t.lower(), ("#555", "#fff"))
    return f'<span class="type-badge" style="background:{bg};color:{fg}">{t}</span>'


def matchup_cell_style(val: float) -> tuple[str, str]:
    for threshold, bg, fg in CELL_BG:
        if val >= threshold:
            return bg, fg
    return "#2d3436", "#fff"


def render_matchup_table(matrix_data: dict) -> str:
    your_team = matrix_data["your_team"]
    opp_team = matrix_data["opp_team"]
    matrix = matrix_data["matrix"]

    header_cells = "".join(
        f'<th style="max-width:90px;overflow:hidden;text-overflow:ellipsis">{m}</th>'
        for m in opp_team
    )
    header = f"<tr><th></th>{header_cells}</tr>"

    rows = ""
    for ym in your_team:
        row = f'<tr><td class="matchup-row-label">{ym}</td>'
        for om in opp_team:
            val = matrix.get(ym, {}).get(om, 1.0)
            bg, fg = matchup_cell_style(val)
            row += f'<td style="background:{bg};color:{fg}">{val:.1f}×</td>'
        row += "</tr>"
        rows += row

    return f'<table class="matchup-table">{header}{rows}</table>'


def team_hash(your_team: list[str], opp_team: list[str]) -> str:
    key = "|".join(your_team) + "||" + "|".join(opp_team)
    return hashlib.md5(key.encode()).hexdigest()


# ─── Cached resource loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PokéCoach model…")
def load_resources():
    from pokecoach.api import get_legal_pool
    return get_legal_pool()


legal_pool = load_resources()

# ─── Session state ─────────────────────────────────────────────────────────────
for key in ("your_team", "opp_team"):
    if key not in st.session_state:
        st.session_state[key] = []


def add_to_team(key: str, mon: str) -> None:
    lst = st.session_state[key]
    if mon and mon not in lst and len(lst) < 6:
        lst.append(mon)


def remove_from_team(key: str, mon: str) -> None:
    lst = st.session_state[key]
    if mon in lst:
        lst.remove(mon)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    from pokecoach.simulation import check_showdown_running
    showdown_ok = check_showdown_running()
    if showdown_ok:
        st.success("🟢 Showdown Running")
    else:
        st.warning("🔴 Showdown Offline")

    c1, c2 = st.columns(2)
    if c1.button("Setup", help="Run scripts/setup_showdown.sh once to clone Showdown"):
        import subprocess
        subprocess.Popen(["bash", "scripts/setup_showdown.sh"])
        st.info("Setup started in background…")
    if c2.button("Start", help="Launch the local Showdown server"):
        import subprocess
        proc = subprocess.Popen(["bash", "scripts/run_showdown.sh"])
        st.session_state["showdown_pid"] = proc.pid
        st.success(f"Started (PID {proc.pid})")

    st.markdown("---")
    st.markdown("**Recommendation Style**")
    mode = st.radio(
        "mode",
        options=["Balanced", "Counter-heavy", "Meta-focused", "Synergy-first", "Custom"],
        index=0,
        label_visibility="collapsed",
    )

    if mode == "Custom":
        st.markdown("*Custom weights:*")
        w_synergy = st.slider("Synergy (team composition)", 0.0, 1.0, 0.45, 0.05)
        w_counter = st.slider("Counter coverage (vs opponent)", 0.0, 1.0, 0.30, 0.05)
        w_viability = st.slider("Viability (meta usage)", 0.0, 1.0, 0.25, 0.05)
        w_content = st.slider("Content (stat similarity)", 0.0, 1.0, 0.15, 0.05)
        weight_overrides = {
            "synergy": w_synergy,
            "counter": w_counter,
            "viability": w_viability,
            "content": w_content,
        }
    else:
        weight_overrides = MODE_WEIGHTS[mode]
        w = weight_overrides
        st.caption(
            f"Synergy {int(w['synergy']*100)}% · Counter {int(w['counter']*100)}% "
            f"· Viability {int(w['viability']*100)}% · Content {int(w['content']*100)}%"
        )

    st.markdown("---")
    st.markdown("**Type Filter**")
    type_prefs = st.multiselect(
        "Require a specific type",
        options=ALL_TYPES,
        default=[],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("PokéCoach v0.3 · RegG VGC 2024")

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚔️ PokéCoach")
st.markdown("Build your VGC team with AI-powered recommendations.")
st.markdown("---")

# ─── 2-Column Layout ───────────────────────────────────────────────────────────
col_left, col_right = st.columns([38, 62], gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Inputs
# ═══════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── STEP 1: YOUR TEAM ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Step 1 — Your Team</div>', unsafe_allow_html=True)
    your_team = st.session_state.your_team

    for mon in list(your_team):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{mon}**")
        if c2.button("✕", key=f"rm_your_{mon}"):
            remove_from_team("your_team", mon)
            st.rerun()

    for _ in range(6 - len(your_team)):
        st.markdown('<span class="slot-empty">— empty slot —</span>', unsafe_allow_html=True)

    if len(your_team) < 6:
        available = [p for p in legal_pool if p not in your_team and p not in st.session_state.opp_team]
        chosen = st.selectbox(
            "Add Pokémon",
            options=[""] + available,
            key="add_your_select",
            label_visibility="collapsed",
        )
        if chosen:
            add_to_team("your_team", chosen)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 2: OPPONENT ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Step 2 — Opponent '
        '<span style="font-size:0.65rem;color:#55657a">(optional)</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<span style="font-size:0.7rem;color:#8899aa">Common archetypes:</span>', unsafe_allow_html=True)
    preset_cols = st.columns(3)
    for i, (name, team) in enumerate(OPPONENT_PRESETS.items()):
        with preset_cols[i % 3]:
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                st.session_state.opp_team = list(team)
                st.rerun()

    opp_team = st.session_state.opp_team

    for mon in list(opp_team):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{mon}**")
        if c2.button("✕", key=f"rm_opp_{mon}"):
            remove_from_team("opp_team", mon)
            st.rerun()

    for _ in range(6 - len(opp_team)):
        st.markdown('<span class="slot-empty">— unknown —</span>', unsafe_allow_html=True)

    if len(opp_team) < 6:
        available_opp = [p for p in legal_pool if p not in opp_team and p not in st.session_state.your_team]
        chosen_opp = st.selectbox(
            "Add opponent Pokémon",
            options=[""] + available_opp,
            key="add_opp_select",
            label_visibility="collapsed",
        )
        if chosen_opp:
            add_to_team("opp_team", chosen_opp)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CTA ────────────────────────────────────────────────────────────────────
    n_slots = max(0, 6 - len(st.session_state.your_team))
    if n_slots == 0:
        st.info("Your team is full! Remove a Pokémon to get suggestions.")
        recommend_clicked = False
    else:
        recommend_clicked = st.button(
            f"⚡ Get Recommendations  ({n_slots} slot{'s' if n_slots != 1 else ''} to fill)",
            type="primary",
            use_container_width=True,
        )

    if recommend_clicked:
        with st.spinner("Calculating recommendations…"):
            from pokecoach.api import get_matchup_matrix, get_win_probability, recommend_team

            result = recommend_team(
                partial_team=st.session_state.your_team,
                opponent_context=st.session_state.opp_team or None,
                type_preferences=type_prefs or None,
                weight_overrides=weight_overrides,
                n_recommendations=n_slots,
            )
            st.session_state["last_result"] = result
            st.session_state["last_result_hash"] = team_hash(
                st.session_state.your_team, st.session_state.opp_team
            )

            if st.session_state.opp_team:
                all_your = st.session_state.your_team + [r["pokemon"] for r in result["recommendations"]]
                st.session_state["last_win_prob"] = get_win_probability(all_your, st.session_state.opp_team)
                st.session_state["last_matchup"] = get_matchup_matrix(all_your, st.session_state.opp_team)
            else:
                st.session_state.pop("last_win_prob", None)
                st.session_state.pop("last_matchup", None)
            st.session_state.pop("last_battle_sim", None)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Outputs
# ═══════════════════════════════════════════════════════════════════════════════
with col_right:
    has_results = "last_result" in st.session_state

    if not has_results:
        # ── ONBOARDING STATE ───────────────────────────────────────────────────
        metrics_note = ""
        try:
            metrics_path = Path("reports/offline_metrics.csv")
            if metrics_path.exists():
                import csv
                with open(metrics_path) as f:
                    for row in csv.DictReader(f):
                        if "hit_rate_5" in row:
                            hr = float(row["hit_rate_5"])
                            random_baseline = 5 / max(1, len(legal_pool))
                            metrics_note = (
                                f"Hit Rate@5: {hr:.1%} vs random baseline ~{random_baseline:.1%} "
                                f"— picks are driven by real co-occurrence patterns."
                            )
                            break
        except Exception:
            pass

        st.markdown(
            """
<div class="onboarding-card">
  <h3>How PokéCoach Works</h3>
  <div class="onboarding-step">
    <span class="step-num">1</span>
    <span class="step-text">Add your existing Pokémon to <b>Your Team</b> on the left.</span>
  </div>
  <div class="onboarding-step">
    <span class="step-num">2</span>
    <span class="step-text">Set an opponent archetype using a quick preset, or leave blank for general picks.</span>
  </div>
  <div class="onboarding-step">
    <span class="step-num">3</span>
    <span class="step-text">Click <b>Get Recommendations</b> — PokéCoach will fill your empty slots.</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        info_text = "📊 Trained on Smogon RegG 2024 data (137 tournament-legal Pokémon)\n\n"
        if metrics_note:
            info_text += f"🎯 {metrics_note}"
        else:
            info_text += (
                "🎯 The hybrid recommender analyses real tournament team co-occurrence data "
                "— picks are not random."
            )
        st.info(info_text)

    else:
        # ── STALE CHECK ────────────────────────────────────────────────────────
        current_hash = team_hash(st.session_state.your_team, st.session_state.opp_team)
        is_stale = current_hash != st.session_state.get("last_result_hash", current_hash)

        if is_stale:
            st.markdown(
                '<div class="stale-banner">⚠️ Your team has changed — click <b>Get Recommendations</b> to update.</div>',
                unsafe_allow_html=True,
            )

        stale_attr = ' style="opacity:0.65"' if is_stale else ""

        # ── RECOMMENDATION CARDS ───────────────────────────────────────────────
        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

        recs = st.session_state["last_result"]["recommendations"]
        partial_team = st.session_state["last_result"]["partial_team"]

        if not recs:
            st.warning("No recommendations found with the current filters.")
        else:
            for i, rec in enumerate(recs):
                mon = rec["pokemon"]
                types = rec["types"]
                moves = rec["moves"]
                scores = rec.get("scores", {})
                synergy_partners = rec.get("synergy_partners", [])
                beats = rec.get("beats", [])

                badges = "".join(type_badge(t) for t in types) if types else ""
                moves_html = " · ".join(moves) if moves else "<em style='color:#55657a'>No moveset data</em>"

                # Plain-English reasons
                reasons = []
                if synergy_partners and partial_team:
                    best_partner, best_val = synergy_partners[0]
                    reasons.append(
                        f"Appears alongside {best_partner} on {int(best_val * 100)}% of tournament teams"
                    )
                if beats:
                    beats_str = ", ".join(f"{p} (×{v:.0f})" for p, v in beats[:3])
                    reasons.append(f"Counters {len(beats)} opponent Pokémon — {beats_str}")
                viability_pct = int(scores.get("viability", 0) * 100)
                if viability_pct > 0:
                    reasons.append(f"Used on {viability_pct}% of competitive teams")

                reasons_html = (
                    "<ul class='reason-list'>"
                    + "".join(f'<li><span class="reason-check">✓</span>{r}</li>' for r in reasons)
                    + "</ul>"
                ) if reasons else ""

                card = f"""
<div class="rec-card"{stale_attr}>
  <div class="rank-label">#{i+1} of {len(legal_pool)} candidates</div>
  <h4>{mon}</h4>
  <div class="types">{badges}</div>
  {reasons_html}
  <div style="color:#a8b8cc;font-size:0.82rem;margin-top:4px">
    <span style="color:#5a7a9a;font-size:0.7rem">Moves: </span>{moves_html}
  </div>
</div>
"""
                st.markdown(card, unsafe_allow_html=True)

        # ── MATCHUP MATRIX (always visible) ───────────────────────────────────
        if "last_matchup" in st.session_state:
            mat = st.session_state["last_matchup"]
            if mat["your_team"] and mat["opp_team"]:
                st.markdown(
                    '<div class="section-title" style="margin-top:16px">Matchup Matrix</div>',
                    unsafe_allow_html=True,
                )
                legend_items = [
                    ("#27ae60", "4× super effective"),
                    ("#6ab04c", "2× effective"),
                    ("#778ca3", "1× neutral"),
                    ("#e67e22", "0.5× resisted"),
                    ("#e74c3c", "0.25× double resisted"),
                ]
                legend_html = "".join(
                    f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
                    f'font-size:0.7rem;color:#a8b8cc">'
                    f'<span style="display:inline-block;width:12px;height:12px;background:{bg};'
                    f'border-radius:2px;margin-right:4px"></span>'
                    f"{label}</span>"
                    for bg, label in legend_items
                )
                st.markdown(
                    f'<div{stale_attr}>'
                    f'<div style="margin-bottom:8px">{legend_html}</div>'
                    + render_matchup_table(mat)
                    + "</div>",
                    unsafe_allow_html=True,
                )

        # ── WIN PROBABILITY ────────────────────────────────────────────────────
        if "last_win_prob" in st.session_state:
            wp = st.session_state["last_win_prob"]
            pct = int(wp["win_probability"] * 100)
            your_cov = wp["your_coverage"]
            opp_cov = wp["opp_coverage"]
            color = "#27ae60" if pct >= 55 else "#e67e22" if pct >= 45 else "#e74c3c"

            opp_names = st.session_state.opp_team
            opp_label = ", ".join(opp_names[:2]) + ("…" if len(opp_names) > 2 else "")

            st.markdown(
                '<div class="section-title" style="margin-top:16px">Win Probability</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
<div class="win-banner"{stale_attr}>
  <div style="font-size:0.72rem;color:#5a7a9a;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px">
    MATCHUP vs {opp_label} — type-coverage estimate
  </div>
  <div class="win-formula">
Your team hits opponent at: {your_cov:.2f}× average<br>
Opponent hits your team at: {opp_cov:.2f}× average<br>
────────────────────────────────<br>
Estimated win rate: <span style="color:{color};font-weight:bold">{pct}%</span>
  </div>
  <div style="font-size:0.72rem;color:#55657a;margin-top:8px">
    This is a type-effectiveness estimate. Run simulated battles for a gameplay-accurate result.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            if st.button(
                "⚔️ Run 100 simulated battles",
                disabled=not showdown_ok,
                help=(
                    "Requires Showdown server. Takes ~2 minutes."
                    if showdown_ok
                    else "Start Showdown server first (sidebar)."
                ),
            ):
                with st.spinner("Running 100 battles… this takes ~2 minutes."):
                    from pokecoach.api import run_battle_sim

                    all_your = st.session_state.your_team + [
                        r["pokemon"]
                        for r in st.session_state.get("last_result", {}).get("recommendations", [])
                    ]
                    sim = run_battle_sim(all_your, st.session_state.opp_team)
                    st.session_state["last_battle_sim"] = sim

            if "last_battle_sim" in st.session_state:
                sim = st.session_state["last_battle_sim"]
                if sim["used_simulation"]:
                    result_text = (
                        f"Simulated result: {sim['wins']}W–{sim['total'] - sim['wins']}L "
                        f"({int(sim['win_rate'] * 100)}%)"
                    )
                else:
                    result_text = f"Analytical: {int(sim['win_rate'] * 100)}% (Showdown offline)"
                st.markdown(
                    f'<div style="color:#a8b8cc;font-size:0.9rem;margin-top:4px">⚔️ {result_text}</div>',
                    unsafe_allow_html=True,
                )

# ─── Footer metrics ─────────────────────────────────────────────────────────────
st.markdown("---")
fc1, fc2, fc3, fc4 = st.columns(4)
fc1.metric("Legal Pool Size", len(legal_pool))
fc2.metric("Your Team", f"{len(st.session_state.your_team)} / 6")
fc3.metric("Slots to Fill", max(0, 6 - len(st.session_state.your_team)))
fc4.metric("Showdown", "Online" if showdown_ok else "Offline")
