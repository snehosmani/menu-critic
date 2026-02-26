import streamlit as st

from ui_theme import inject_ui_theme, render_hero, render_info_cards, render_sidebar_nav, section_heading


st.set_page_config(page_title="About | Menu Critic", page_icon="ℹ️", layout="wide")

inject_ui_theme()
render_sidebar_nav("about")

render_hero(
    title="About Menu Critic",
    kicker="What it does",
    description=(
        "Menu Critic is a demoable AI-native Streamlit app that reviews restaurant menus and suggests "
        "ways to improve conversion, average order value (AOV), and customer experience."
    ),
)

render_info_cards(
    [
        ("Input flexibility", "Works with pasted menu text or uploaded menu images."),
        ("Dual mode", "Serious optimization mode and playful roast mode."),
        ("Structured output", "Returns scores, rewrites, tests, and red flags in strict JSON."),
    ]
)

section_heading("What it does")
st.markdown(
    """
    <div class="mc-card">
      <ul class="mc-list">
        <li>Accepts either pasted menu text or an uploaded menu image (JPG/PNG).</li>
        <li>Runs in two modes: <strong>Fix my menu</strong> (serious, revenue-focused) and <strong>Roast my menu</strong> (funny but not mean).</li>
        <li>Returns a structured critique with scorecard, top changes, revenue levers, rewrite examples, A/B tests, and red flags.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

section_heading("How it works")
st.markdown(
    """
    <div class="mc-card">
      <ol class="mc-list">
        <li>If you upload an image, the app first tries a Groq vision-capable model to extract menu text.</li>
        <li>If image extraction fails or confidence is low, the app asks for pasted text instead.</li>
        <li>The app sends menu text to Groq and requests strict JSON output with a fixed schema.</li>
        <li>The UI validates and renders the JSON into readable sections and lets you download it.</li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

section_heading("Safety and reliability")
st.markdown(
    """
    <div class="mc-card">
      <ul class="mc-list">
        <li>Uses <code>GROQ_API_KEY</code> from Streamlit secrets (never hardcoded).</li>
        <li>Session rate limiting: one request every 10 seconds.</li>
        <li>Input caps for text and images; images are resized/compressed before vision requests.</li>
        <li>Friendly fallback states for rate limits, invalid JSON, and image parsing failures.</li>
        <li>Roast mode is constrained to be playful and non-harassing.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

section_heading("Build note")
st.markdown(
    """
    <div class="mc-card">
      <p class="mc-muted">
        Built via ChatGPT Codex in ~1 hour on 25th Feb, and ideated by yours truly - Snehal.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
