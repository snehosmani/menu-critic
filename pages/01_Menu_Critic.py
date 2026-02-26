from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st
from ui_theme import render_sidebar_nav

from menu_critic_core import (
    MAX_TEXT_CHARS,
    MIN_EXTRACTED_TEXT_CHARS,
    MIN_VISION_CONFIDENCE,
    REQUEST_COOLDOWN_SECONDS,
    InvalidJSONResponse,
    MenuCriticError,
    RateLimitLikeError,
    VisionExtractionError,
    analyze_menu_text,
    clamp_text_input,
    dumps_pretty_json,
    extract_menu_text_from_image,
    get_groq_client,
    preprocess_image_for_groq,
)


ASSETS_DIR = Path("assets")
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

st.set_page_config(page_title="Menu Critic", page_icon="🍽️", layout="wide")


def _inject_styles() -> None:
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
          margin: 0 0 0.3rem;
          font-size: 1rem;
          color: var(--mc-ink);
        }
        .mc-card p {
          margin: 0;
          color: var(--mc-muted);
          font-size: 0.9rem;
        }
        .mc-section {
          margin-top: 0.75rem;
          margin-bottom: 0.1rem;
          font-size: 1.05rem;
          font-weight: 700;
          color: var(--mc-ink);
        }
        div[data-testid="stMetric"] {
          background: rgba(255,255,255,0.96);
          border: 1px solid var(--mc-border);
          border-radius: 14px;
          padding: 0.5rem 0.6rem;
          box-shadow: 0 6px 18px rgba(31,42,46,0.05);
        }
        div[data-testid="stFileUploader"] {
          background: rgba(255,255,255,0.8);
          border-radius: 12px;
          padding: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_gif(name: str, caption: str) -> None:
    path = ASSETS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.caption(f"[Missing asset: `{path}`] {caption}")


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _render_sample_downloads() -> None:
    st.markdown('<div class="mc-section">Sample Inputs</div>', unsafe_allow_html=True)
    st.caption("Use these sample files for demos. You can replace them with your own local assets.")

    text_path = _first_existing_path([Path("assets/sample_menu.txt"), Path("sample_menu.txt")])
    image_path = _first_existing_path(
        [
            Path("assets/sample_menu_image.png"),
            Path("assets/sample_menu_image.jpg"),
            Path("assets/sample_menu_image.jpeg"),
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        if text_path:
            st.download_button(
                "Download Sample Menu Text",
                data=text_path.read_bytes(),
                file_name=text_path.name,
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.caption("Add `sample_menu.txt` (root or `assets/`) to enable text download.")
    with c2:
        if image_path:
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            st.download_button(
                "Download Sample Menu Image",
                data=image_path.read_bytes(),
                file_name=image_path.name,
                mime=mime,
                use_container_width=True,
            )
        else:
            st.caption(
                "Add `assets/sample_menu_image.png` or `assets/sample_menu_image.jpg` to enable image download."
            )


def _init_state() -> None:
    st.session_state.setdefault("last_request_ts", 0.0)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_result_json_text", "")
    st.session_state.setdefault("last_invalid_json_raw", "")
    st.session_state.setdefault("last_invalid_json_error", "")
    st.session_state.setdefault("last_critique_request", None)
    st.session_state.setdefault("queued_retry", False)
    st.session_state.setdefault("last_uploaded_image_bytes", None)
    st.session_state.setdefault("last_uploaded_image_name", None)
    st.session_state.setdefault("last_uploaded_image_mime", None)


def _render_scorecard(scores: dict[str, int]) -> None:
    st.markdown('<div class="mc-section">Scorecard</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clarity", scores["clarity"])
    c2.metric("Pricing Psychology", scores["pricing_psychology"])
    c3.metric("Upsell Potential", scores["upsell_potential"])
    c4.metric("Menu Structure", scores["menu_structure"])
    c5.metric("Dietary Signals", scores["dietary_signals"])


def _render_list_section(title: str, items: list[str]) -> None:
    st.markdown(f'<div class="mc-section">{title}</div>', unsafe_allow_html=True)
    if not items:
        st.write("No items returned.")
        return
    for idx, item in enumerate(items, start=1):
        st.write(f"{idx}. {item}")


def _render_revenue_levers(revenue_levers: dict[str, list[str]]) -> None:
    st.markdown('<div class="mc-section">Revenue Levers</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, key, label in zip(cols, ["conversion", "aov", "margin"], ["Conversion", "AOV", "Margin"]):
        with col:
            st.markdown(f"**{label}**")
            items = revenue_levers.get(key, [])
            if not items:
                st.write("-")
            for item in items:
                st.write(f"- {item}")


def _render_rewrite_examples(items: list[dict[str, str]]) -> None:
    st.markdown('<div class="mc-section">Rewrite Examples</div>', unsafe_allow_html=True)
    if not items:
        st.write("No rewrite examples returned.")
        return
    for idx, item in enumerate(items, start=1):
        with st.container(border=True):
            st.markdown(f"**Example {idx}**")
            st.write(f"**Original:** {item.get('original', '')}")
            st.write(f"**Rewritten:** {item.get('rewritten', '')}")
            st.write(f"**Why it helps:** {item.get('why_it_helps', '')}")


def _render_ab_tests(items: list[dict[str, str]]) -> None:
    st.markdown('<div class="mc-section">A/B Tests</div>', unsafe_allow_html=True)
    if not items:
        st.write("No A/B tests returned.")
        return
    for idx, item in enumerate(items, start=1):
        with st.expander(f"Test {idx}: {item.get('hypothesis', 'Hypothesis')}"):
            st.write(f"**Hypothesis:** {item.get('hypothesis', '')}")
            st.write(f"**Variant A:** {item.get('variant_a', '')}")
            st.write(f"**Variant B:** {item.get('variant_b', '')}")
            st.write(f"**Success metric:** {item.get('success_metric', '')}")


def _render_result(result: dict[str, Any], json_text: str) -> None:
    _render_scorecard(result["scores"])
    _render_list_section("Top 5 Changes", result["top_5_changes"])
    _render_revenue_levers(result["revenue_levers"])
    _render_rewrite_examples(result["rewrite_examples"])
    _render_ab_tests(result["ab_tests"])
    _render_list_section("Red Flags", result["red_flags"])

    st.download_button(
        label="Download JSON",
        data=json_text,
        file_name="menu_critic_output.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_reference_panel(last_request: dict[str, Any]) -> None:
    st.markdown('<div class="mc-section">Reference</div>', unsafe_allow_html=True)
    source = last_request.get("source", "unknown")
    st.caption(f"Input source: {source}")
    if source == "image" and st.session_state.get("last_uploaded_image_bytes"):
        st.image(
            st.session_state["last_uploaded_image_bytes"],
            caption=st.session_state.get("last_uploaded_image_name") or "Uploaded menu image",
            use_container_width=True,
        )
        if last_request.get("vision_confidence") is not None:
            st.caption(f"Vision extraction confidence: {last_request['vision_confidence']:.2f}")
        if last_request.get("vision_notes"):
            st.caption(f"Vision notes: {last_request['vision_notes']}")
        with st.expander("Extracted text used for analysis", expanded=False):
            st.text(last_request.get("menu_text", ""))
    elif source == "text":
        with st.expander("Menu text used for analysis", expanded=False):
            st.text(last_request.get("menu_text", ""))


def _build_critique_request(
    input_mode: str,
    menu_text_input: str,
    uploaded_image: Any,
    mode: str,
    goal: str,
    context: str,
    client: Any,
) -> dict[str, Any]:
    if input_mode == "Paste menu text":
        menu_text = clamp_text_input(menu_text_input)
        if not menu_text:
            logger.info("Analyze blocked: empty pasted text input.")
            raise ValueError("Please paste menu text before analyzing.")
        logger.info(
            "Preparing text-based critique request: mode=%s goal=%s chars=%s context_provided=%s",
            mode,
            goal,
            len(menu_text),
            bool(context.strip()),
        )
        return {
            "menu_text": menu_text,
            "mode": mode,
            "goal": goal,
            "context": context.strip(),
            "source": "text",
        }

    if uploaded_image is None:
        logger.info("Analyze blocked: image mode selected with no uploaded file.")
        raise ValueError("Please upload a menu image before analyzing.")

    image_bytes = uploaded_image.getvalue()
    st.session_state["last_uploaded_image_bytes"] = image_bytes
    st.session_state["last_uploaded_image_name"] = getattr(uploaded_image, "name", "menu_image")
    st.session_state["last_uploaded_image_mime"] = getattr(uploaded_image, "type", None)
    logger.info(
        "Preparing image-based critique request: filename=%s bytes=%s mode=%s goal=%s",
        st.session_state["last_uploaded_image_name"],
        len(image_bytes),
        mode,
        goal,
    )

    data_url, image_meta = preprocess_image_for_groq(uploaded_image)
    vision_result = extract_menu_text_from_image(client, data_url)
    extracted_text = clamp_text_input(vision_result.menu_text)

    if (
        vision_result.confidence < MIN_VISION_CONFIDENCE
        or len(extracted_text.strip()) < MIN_EXTRACTED_TEXT_CHARS
    ):
        msg = (
            "I tried to read that menu image and my OCR-brain briefly turned into alphabet soup. "
            "Please try a clearer photo or paste the menu text instead."
        )
        details = [f"Vision confidence: {vision_result.confidence:.2f}"]
        if vision_result.notes:
            details.append(f"Notes: {vision_result.notes}")
        logger.warning(
            "Vision extraction below threshold. confidence=%.2f chars=%s notes=%s",
            vision_result.confidence,
            len(extracted_text.strip()),
            vision_result.notes,
        )
        raise VisionExtractionError(msg + " " + " | ".join(details), raw_output=vision_result.raw)

    logger.info(
        "Image-based critique request ready: extracted_chars=%s confidence=%.2f meta=%s",
        len(extracted_text),
        vision_result.confidence,
        image_meta,
    )
    return {
        "menu_text": extracted_text,
        "mode": mode,
        "goal": goal,
        "context": context.strip(),
        "source": "image",
        "vision_confidence": vision_result.confidence,
        "vision_notes": vision_result.notes,
        "image_meta": image_meta,
    }


def _enforce_rate_limit() -> None:
    now = time.time()
    delta = now - float(st.session_state.get("last_request_ts", 0.0))
    if delta < REQUEST_COOLDOWN_SECONDS:
        wait = int(REQUEST_COOLDOWN_SECONDS - delta + 0.999)
        logger.info("Session rate limit triggered. wait_seconds=%s", wait)
        raise ValueError(f"Please wait {wait}s before sending another request.")
    st.session_state["last_request_ts"] = now


_init_state()
_inject_styles()
render_sidebar_nav("menu_critic")

st.markdown(
    """
    <div class="mc-hero">
      <div class="mc-kicker">AI-native Menu Optimization</div>
      <h1>Menu Critic</h1>
      <p>Critique restaurant menus for conversion, AOV, and guest experience. Run a serious optimization pass or a playful roast, then export strict JSON for demos, experiments, or workflows.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns(3)
hero_cards = [
    ("Two input modes", "Paste menu text or upload a JPG/PNG menu image."),
    ("Two analysis styles", "Fix mode for revenue. Roast mode for demo-friendly fun."),
    ("Structured output", "Strict JSON scorecards, rewrites, A/B tests, and red flags."),
]
for col, (title, desc) in zip(hero_cols, hero_cards):
    with col:
        st.markdown(
            f'<div class="mc-card"><h3>{title}</h3><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

_render_sample_downloads()

api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    logger.warning("GROQ_API_KEY missing in Streamlit secrets.")
    st.warning(
        "Missing `GROQ_API_KEY` in Streamlit secrets. Add it to `.streamlit/secrets.toml` "
        "(local) or Streamlit Community Cloud secrets, then refresh."
    )
    st.code('GROQ_API_KEY = "your_groq_api_key_here"', language="toml")
    st.stop()

left, right = st.columns([1.2, 0.8], vertical_alignment="top")

with left:
    input_mode = st.radio("Input type", ["Paste menu text", "Upload menu image"], horizontal=True)
    menu_text_input = st.text_area(
        "Menu text",
        height=260,
        placeholder="Paste your menu text here...",
        help=f"Max {MAX_TEXT_CHARS:,} characters.",
        disabled=input_mode != "Paste menu text",
    )
    uploaded_image = st.file_uploader(
        "Menu image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        disabled=input_mode != "Upload menu image",
        help="Use a clear, readable image for best extraction results.",
    )
    if uploaded_image is not None and input_mode == "Upload menu image":
        with st.expander("Preview uploaded image", expanded=False):
            st.image(uploaded_image, caption="Uploaded image preview", width=280)

with right:
    mode = st.radio(
        "Mode",
        ["Fix my menu", "Roast my menu"],
        help="Roast mode is playful and critiques the menu, not people.",
    )
    goal = st.selectbox(
        "Primary goal",
        ["Increase conversion", "Increase AOV", "Improve experience & retention"],
    )
    context = st.text_input(
        "Optional context (cuisine / restaurant type)",
        placeholder="e.g., fast-casual Thai, brunch cafe, pizza delivery",
    )
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

retry_clicked = False
if st.session_state.get("last_invalid_json_raw"):
    retry_clicked = st.button("Retry last analysis", use_container_width=True)

request_to_run = None
queued_retry = bool(st.session_state.get("queued_retry"))
if queued_retry and st.session_state.get("last_critique_request"):
    st.session_state["queued_retry"] = False
    request_to_run = "retry"
elif analyze_clicked:
    request_to_run = "new"
elif retry_clicked and st.session_state.get("last_critique_request"):
    request_to_run = "retry"

if request_to_run:
    try:
        logger.info("Analyze requested. request_type=%s input_mode=%s", request_to_run, input_mode)
        _enforce_rate_limit()
        client = get_groq_client(api_key)

        spinner_message = "Analyzing your menu..."
        if request_to_run == "new" and input_mode == "Upload menu image":
            spinner_message = "Reading image, extracting menu text, and analyzing..."
        elif request_to_run == "retry":
            spinner_message = "Retrying analysis..."

        with st.spinner(spinner_message):
            if request_to_run == "new":
                critique_request = _build_critique_request(
                    input_mode=input_mode,
                    menu_text_input=menu_text_input,
                    uploaded_image=uploaded_image,
                    mode=mode,
                    goal=goal,
                    context=context,
                    client=client,
                )
                st.session_state["last_critique_request"] = critique_request
                st.session_state["last_invalid_json_raw"] = ""
                st.session_state["last_invalid_json_error"] = ""
                logger.info("Stored new critique request in session. source=%s", critique_request.get("source"))
            else:
                critique_request = st.session_state["last_critique_request"]
                logger.info("Retrying previous critique request. source=%s", critique_request.get("source"))

            result, _raw_json = analyze_menu_text(
                client=client,
                menu_text=critique_request["menu_text"],
                mode=critique_request["mode"],
                goal=critique_request["goal"],
                context=critique_request.get("context"),
            )

        st.session_state["last_result"] = result
        st.session_state["last_result_json_text"] = dumps_pretty_json(result)
        st.session_state["last_invalid_json_raw"] = ""
        st.session_state["last_invalid_json_error"] = ""
        logger.info("Analysis succeeded and result saved to session.")

    except ValueError as exc:
        logger.info("Validation/user input error during analyze: %s", exc)
        st.error(str(exc))
    except VisionExtractionError as exc:
        logger.warning("Vision extraction error shown to user: %s", exc)
        st.warning("Image parsing hiccup: I roasted myself before roasting your menu.")
        st.write(
            "Oops. I looked at that menu image and confidently extracted chaos. "
            "Please try a clearer image or paste the menu text instead."
        )
        _show_gif("confused.gif", "Confused but trying.")
        st.caption(str(exc))
    except InvalidJSONResponse as exc:
        logger.warning("Invalid JSON response from model. error=%s raw_len=%s", exc, len(exc.raw_output or ""))
        st.session_state["last_invalid_json_raw"] = exc.raw_output
        st.session_state["last_invalid_json_error"] = str(exc)
        st.error("The model replied, but the JSON was malformed or didn't match the schema.")
        _show_gif("this_is_fine.gif", "This is fine.")
        st.caption(str(exc))
        st.text_area("Raw model output", exc.raw_output, height=220)
        if st.button("Retry", key="retry_invalid_inline"):
            logger.info("User requested inline retry after invalid JSON.")
            st.session_state["queued_retry"] = True
            st.rerun()
    except RateLimitLikeError:
        logger.warning("Groq rate limit-like error shown to user.")
        st.error("Groq is taking a nap - try again in a minute.")
        _show_gif("sad.gif", "API nap time.")
    except MenuCriticError as exc:
        logger.exception("MenuCriticError during analyze flow.")
        st.error(f"Request failed: {exc}")
        _show_gif("sad.gif", "Something went sideways.")
    except Exception as exc:
        logger.exception("Unexpected error during analyze flow.")
        st.error(f"Unexpected error: {exc}")
        _show_gif("sad.gif", "Unexpected plot twist.")

last_result = st.session_state.get("last_result")
if last_result:
    last_request = st.session_state.get("last_critique_request") or {}
    results_col, ref_col = st.columns([1.6, 0.9], vertical_alignment="top")
    with results_col:
        _render_result(last_result, st.session_state["last_result_json_text"])
    with ref_col:
        _render_reference_panel(last_request)

if st.session_state.get("last_invalid_json_raw") and not request_to_run:
    logger.info("Showing persisted invalid JSON raw output panel.")
    st.warning("Last run returned invalid JSON. You can retry using the button above.")
    st.text_area("Last raw model output", st.session_state["last_invalid_json_raw"], height=220)
