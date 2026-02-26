import streamlit as st

from ui_theme import inject_ui_theme, render_hero, render_info_cards, render_sidebar_nav, section_heading


st.set_page_config(page_title="Why I Built This | Menu Critic", page_icon="💼", layout="wide")

inject_ui_theme()
render_sidebar_nav("why")

render_hero(
    title="Why I Built This",
    kicker="Square / Block",
    description=(
        "This is a product-thinking demo for a Square/Block-style role: solve a real merchant pain point, "
        "use AI where it helps, and ship a workflow that is easy to demo."
    ),
)

render_info_cards(
    [
        ("Merchant pain", "Great food still loses orders when menus are unclear or undersell value."),
        ("AI fit", "Vision extraction + structured critique compresses manual review work."),
        ("Product angle", "Turns feedback into experiments and rewrites that can be measured."),
    ]
)

section_heading("The pitch")
st.markdown(
    """
    <div class="mc-card">
      <p class="mc-muted">
        Restaurants often have great food and still lose orders because their menu is hard to scan, badly structured,
        or leaves money on the table with weak upsell cues. Menu Critic turns a menu into a concrete action plan plus test ideas.
      </p>
      <ul class="mc-list" style="margin-top:0.65rem;">
        <li><strong>Merchant-friendly</strong>: simple inputs, clear outputs</li>
        <li><strong>AI-native</strong>: image extraction plus structured critique</li>
        <li><strong>Demo-ready</strong>: serious mode and roast mode</li>
        <li><strong>Deployable</strong>: Streamlit Community Cloud plus secrets</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

section_heading("Why this fits Square / Block")
st.markdown(
    """
    <div class="mc-card">
      <ul class="mc-list">
        <li>Merchant outcome focused: conversion, AOV, retention</li>
        <li>Practical seller-tooling workflow: menu content to experiments</li>
        <li>Produces measurable next steps (A/B tests and rewrite examples)</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

section_heading("Links")
st.markdown(
    """
    <div class="mc-card">
      <ul class="mc-list">
        <li>GitHub: <a href="https://github.com/snehosmani" target="_blank">https://github.com/snehosmani</a></li>
        <li>LinkedIn: <a href="https://www.linkedin.com/in/snehalhosmani/" target="_blank">https://www.linkedin.com/in/snehalhosmani/</a></li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
