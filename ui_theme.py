import streamlit as st


def inject_ui_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --mc-bg: #f7f4ee;
          --mc-ink: #1f2a2e;
          --mc-card: #ffffff;
          --mc-accent: #d96941;
          --mc-accent-2: #1f7a6a;
          --mc-muted: #667085;
          --mc-border: rgba(31, 42, 46, 0.08);
          --mc-shadow: 0 14px 30px rgba(31,42,46,0.08);
        }
        .stApp {
          background:
            radial-gradient(circle at 8% 8%, rgba(217,105,65,0.10), transparent 35%),
            radial-gradient(circle at 92% 14%, rgba(31,122,106,0.10), transparent 38%),
            linear-gradient(180deg, #fbfaf7 0%, var(--mc-bg) 100%);
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] ul li:first-child {
          display: none;
        }
        .mc-hero {
          background: linear-gradient(135deg, #fff, #fff7ef 55%, #eefaf7);
          border: 1px solid var(--mc-border);
          border-radius: 18px;
          padding: 1.1rem 1.2rem;
          box-shadow: var(--mc-shadow);
          margin-bottom: 1rem;
        }
        .mc-kicker {
          display: inline-block;
          background: rgba(217,105,65,0.12);
          color: #a34728;
          border: 1px solid rgba(217,105,65,0.18);
          border-radius: 999px;
          padding: 0.2rem 0.65rem;
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.02em;
          margin-bottom: 0.4rem;
        }
        .mc-hero h1 {
          margin: 0;
          font-size: 2rem;
          line-height: 1.05;
          color: var(--mc-ink);
        }
        .mc-hero p {
          margin: 0.5rem 0 0;
          color: var(--mc-muted);
          font-size: 0.97rem;
        }
        .mc-card {
          background: var(--mc-card);
          border: 1px solid var(--mc-border);
          border-radius: 16px;
          padding: 0.9rem 1rem;
          box-shadow: var(--mc-shadow);
          margin-bottom: 0.8rem;
        }
        .mc-card h3 {
          margin: 0 0 0.35rem;
          font-size: 1rem;
          color: var(--mc-ink);
        }
        .mc-card p {
          margin: 0;
          color: var(--mc-muted);
          font-size: 0.92rem;
        }
        .mc-section {
          margin-top: 0.75rem;
          margin-bottom: 0.4rem;
          font-size: 1.05rem;
          font-weight: 700;
          color: var(--mc-ink);
        }
        .mc-list {
          margin: 0;
          padding-left: 1.05rem;
          color: var(--mc-ink);
          line-height: 1.55;
        }
        .mc-list li { margin-bottom: 0.35rem; }
        .mc-muted { color: var(--mc-muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, description: str, kicker: str) -> None:
    st.markdown(
        f"""
        <div class="mc-hero">
          <div class="mc-kicker">{kicker}</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_cards(cards: list[tuple[str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="mc-card"><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )


def section_heading(title: str) -> None:
    st.markdown(f'<div class="mc-section">{title}</div>', unsafe_allow_html=True)


def card_markdown(markdown_text: str) -> None:
    st.markdown(f'<div class="mc-card">{markdown_text}</div>', unsafe_allow_html=True)


def render_sidebar_nav(active_page: str) -> None:
    with st.sidebar:
        st.markdown("### Menu Critic")
        st.caption("AI-native menu review")
        st.page_link(
            "pages/01_Menu_Critic.py",
            label="Menu Critic",
            icon="🍽️",
            disabled=active_page == "menu_critic",
        )
        st.page_link(
            "pages/02_About.py",
            label="About",
            icon="ℹ️",
            disabled=active_page == "about",
        )
        st.page_link(
            "pages/03_Why_I_Built_This.py",
            label="Why I Built This",
            icon="💼",
            disabled=active_page == "why",
        )
