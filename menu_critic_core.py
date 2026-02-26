from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from groq import Groq
from PIL import Image

logger = logging.getLogger(__name__)


TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
REQUEST_COOLDOWN_SECONDS = 10
MAX_TEXT_CHARS = 12_000
MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024
TARGET_IMAGE_BYTES = 3_500_000
MIN_EXTRACTED_TEXT_CHARS = 20
MIN_VISION_CONFIDENCE = 0.45

SCORE_KEYS = [
    "clarity",
    "pricing_psychology",
    "upsell_potential",
    "menu_structure",
    "dietary_signals",
]

CRITIQUE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "scores",
        "top_5_changes",
        "revenue_levers",
        "rewrite_examples",
        "ab_tests",
        "red_flags",
    ],
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "required": SCORE_KEYS,
            "properties": {
                "clarity": {"type": "integer", "minimum": 0, "maximum": 100},
                "pricing_psychology": {"type": "integer", "minimum": 0, "maximum": 100},
                "upsell_potential": {"type": "integer", "minimum": 0, "maximum": 100},
                "menu_structure": {"type": "integer", "minimum": 0, "maximum": 100},
                "dietary_signals": {"type": "integer", "minimum": 0, "maximum": 100},
            },
        },
        "top_5_changes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "revenue_levers": {
            "type": "object",
            "additionalProperties": False,
            "required": ["conversion", "aov", "margin"],
            "properties": {
                "conversion": {"type": "array", "items": {"type": "string"}},
                "aov": {"type": "array", "items": {"type": "string"}},
                "margin": {"type": "array", "items": {"type": "string"}},
            },
        },
        "rewrite_examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["original", "rewritten", "why_it_helps"],
                "properties": {
                    "original": {"type": "string"},
                    "rewritten": {"type": "string"},
                    "why_it_helps": {"type": "string"},
                },
            },
        },
        "ab_tests": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "hypothesis",
                    "variant_a",
                    "variant_b",
                    "success_metric",
                ],
                "properties": {
                    "hypothesis": {"type": "string"},
                    "variant_a": {"type": "string"},
                    "variant_b": {"type": "string"},
                    "success_metric": {"type": "string"},
                },
            },
        },
        "red_flags": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class VisionExtractionResult:
    menu_text: str
    confidence: float
    notes: str = ""
    raw: str | None = None
    usage: dict[str, int | None] | None = None
    model: str | None = None


class MenuCriticError(Exception):
    pass


class GroqSetupError(MenuCriticError):
    pass


class InvalidJSONResponse(MenuCriticError):
    def __init__(self, raw_output: str, message: str = "Model returned invalid JSON") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class VisionExtractionError(MenuCriticError):
    def __init__(self, message: str, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class RateLimitLikeError(MenuCriticError):
    pass


class SuspiciousMenuInputError(MenuCriticError):
    pass


def _usage_from_response(resp: Any) -> dict[str, int | None]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def get_groq_client(api_key: str | None) -> Groq:
    if not api_key:
        logger.warning("Groq client setup failed: missing GROQ_API_KEY.")
        raise GroqSetupError("Missing GROQ_API_KEY")
    logger.debug("Initializing Groq client.")
    return Groq(api_key=api_key)


def clamp_text_input(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) > MAX_TEXT_CHARS:
        logger.info("Clamping menu text input from %s to %s characters.", len(cleaned), MAX_TEXT_CHARS)
        cleaned = cleaned[:MAX_TEXT_CHARS]
    return cleaned


def validate_menu_like_text(text: str, source: str = "text") -> None:
    candidate = (text or "").strip()
    lowered = candidate.lower()

    if len(candidate) < 20:
        raise SuspiciousMenuInputError(
            f"That {source} input is too short to look like a real menu. Nice try though."
        )

    tokens = re.findall(r"[A-Za-z]{2,}", candidate)
    if not tokens:
        raise SuspiciousMenuInputError(
            f"That {source} input does not contain readable menu text."
        )

    alpha_chars = [c for c in candidate if c.isalpha()]
    vowels = sum(1 for c in alpha_chars if c.lower() in "aeiou")
    vowel_ratio = (vowels / len(alpha_chars)) if alpha_chars else 0.0
    long_token_ratio = (
        sum(1 for t in tokens if len(t) >= 9) / len(tokens)
        if tokens
        else 1.0
    )
    has_price = bool(re.search(r"[$€£]\s?\d|\b\d{1,3}\.\d{2}\b", candidate))
    has_line_breaks = candidate.count("\n") >= 2
    has_menu_words = any(
        word in lowered
        for word in [
            "menu",
            "burger",
            "pizza",
            "salad",
            "drink",
            "appetizer",
            "dessert",
            "chicken",
            "fries",
            "soup",
            "sandwich",
            "pasta",
            "rice",
            "combo",
            "add on",
            "addons",
        ]
    )

    # Catch obvious keyboard-smash / gibberish inputs like "dfdsfsdg".
    if not has_price and not has_line_breaks and not has_menu_words:
        if len(tokens) <= 3 and (vowel_ratio < 0.22 or long_token_ratio > 0.6):
            logger.info(
                "Rejected suspicious %s input as non-menu text. chars=%s tokens=%s vowel_ratio=%.2f",
                source,
                len(candidate),
                len(tokens),
                vowel_ratio,
            )
            raise SuspiciousMenuInputError(
                f"That {source} input does not look like a menu. It looks more like a keyboard warm-up."
            )

    # For longer text, require at least one menu-ish signal.
    if len(candidate) < 120 and not (has_price or has_line_breaks or has_menu_words):
        logger.info(
            "Rejected suspicious %s input due to missing menu signals. chars=%s tokens=%s",
            source,
            len(candidate),
            len(tokens),
        )
        raise SuspiciousMenuInputError(
            f"That {source} input doesn't look menu-ish yet. Paste actual menu items or upload a clearer menu image."
        )


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.getchannel("A"))
        return base
    if image.mode == "P":
        return image.convert("RGB")
    return image.convert("RGB")


def preprocess_image_for_groq(uploaded_file: Any) -> tuple[str, dict[str, Any]]:
    if uploaded_file is None:
        logger.warning("Image preprocessing attempted without an uploaded file.")
        raise ValueError("No image uploaded.")
    if getattr(uploaded_file, "size", 0) > MAX_IMAGE_UPLOAD_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        logger.warning("Rejected image upload: size=%s bytes exceeds limit=%s.", uploaded_file.size, MAX_IMAGE_UPLOAD_BYTES)
        raise ValueError(
            f"Image is too large ({size_mb:.1f} MB). Please upload an image under "
            f"{MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    logger.info("Preprocessing image upload: name=%s size=%s bytes", getattr(uploaded_file, "name", None), getattr(uploaded_file, "size", None))
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = _to_rgb(image)
    image.thumbnail((1600, 1600))

    buffer = io.BytesIO()
    jpeg_bytes = b""
    used_quality = 85
    for quality in [90, 85, 78, 72, 65, 58, 50]:
        buffer.seek(0)
        buffer.truncate(0)
        image.save(buffer, format="JPEG", optimize=True, quality=quality)
        candidate = buffer.getvalue()
        jpeg_bytes = candidate
        used_quality = quality
        if len(candidate) <= TARGET_IMAGE_BYTES:
            break

    if len(jpeg_bytes) > TARGET_IMAGE_BYTES:
        logger.warning(
            "Image remained too large after compression: %s bytes (target %s).",
            len(jpeg_bytes),
            TARGET_IMAGE_BYTES,
        )
        raise ValueError(
            "Image is still too large after resize/compression. Try a smaller/cropped image "
            "or paste menu text instead."
        )

    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    meta = {
        "width": image.width,
        "height": image.height,
        "bytes": len(jpeg_bytes),
        "quality": used_quality,
    }
    logger.info("Image preprocessing complete: %s", meta)
    return data_url, meta


def _chat_content_text(resp: Any) -> str:
    return (resp.choices[0].message.content or "").strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


def extract_menu_text_from_image(client: Groq, image_data_url: str) -> VisionExtractionResult:
    system_prompt = (
        "You extract restaurant menu text from images. Output ONLY valid JSON in English. "
        "Do not add markdown fences."
    )
    user_prompt = (
        "Read the menu image and extract the visible menu text in English.\n"
        "Return JSON with keys: menu_text (string), confidence (number 0 to 1), notes (string).\n"
        "If the image is blurry, obstructed, or not a menu, set confidence below 0.45 and explain in notes.\n"
        "Preserve line breaks where possible."
    )
    try:
        logger.info("Sending Groq vision extraction request using model=%s.", VISION_MODEL)
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        )
    except Exception as exc:
        if _is_rate_limit_error(exc):
            logger.warning("Vision extraction hit rate limit-like error: %s", exc)
            raise RateLimitLikeError(str(exc)) from exc
        logger.exception("Vision extraction request failed.")
        raise VisionExtractionError(f"Vision request failed: {exc}") from exc

    raw = _chat_content_text(resp)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Vision extraction returned invalid JSON. raw_len=%s", len(raw))
        raise VisionExtractionError("Vision response was not valid JSON.", raw_output=raw) from exc

    menu_text = str(data.get("menu_text", "")).strip()
    notes = str(data.get("notes", "")).strip()
    confidence_raw = data.get("confidence", 0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    logger.info(
        "Vision extraction complete: confidence=%.2f menu_text_chars=%s",
        confidence,
        len(menu_text),
    )

    return VisionExtractionResult(
        menu_text=menu_text,
        confidence=confidence,
        notes=notes,
        raw=raw,
        usage=_usage_from_response(resp),
        model=VISION_MODEL,
    )


def _critique_system_prompt() -> str:
    return (
        "You are Menu Critic, an expert in restaurant menu conversion optimization, average order value, "
        "and customer experience. Always respond in English and output JSON only (no markdown).\n"
        "If mode is Roast, be funny and specific but never cruel. Roast the menu copy/layout/pricing choices, "
        "not the owner or any people. No harassment, slurs, or personal attacks.\n"
        "In Roast mode, use sharper humor, vivid metaphors, playful one-liners, and consultant-style sarcasm "
        "while still being actionable.\n"
        "Avoid bland corporate wording in Roast mode. Each major point should sound like a real roast, not a polite audit.\n"
        "Focus on practical, testable improvements."
    )


def _critique_user_prompt(menu_text: str, mode: str, goal: str, context: str | None) -> str:
    safe_context = (context or "").strip() or "Not provided"
    is_roast = mode.lower().startswith("roast")
    mode_specific = (
        "Roast style requirements:\n"
        "- Make the critique genuinely funny and specific (not generic, not mild).\n"
        "- Roast the menu writing/structure/pricing like a witty consultant doing stand-up with receipts.\n"
        "- Every joke must still include a useful fix.\n"
        "- Keep it playful, not cruel, and never target people.\n"
        "- `top_5_changes` and `red_flags` should read like punchy roasts with actionable advice.\n"
        "- Prefer lines that combine a roast + fix in one sentence.\n"
        "- Use colorful phrasing (examples of tone only): 'reads like a tax form', 'buried like a secret menu witness', "
        "'priced like it includes a side of rent'.\n"
        "- Do not overuse the same joke structure.\n"
        "- `rewrite_examples[].why_it_helps` should keep a witty tone while explaining the conversion logic.\n"
        "- `ab_tests[].hypothesis` can be playful, but `success_metric` must stay practical.\n\n"
    ) if is_roast else (
        "Fix mode requirements:\n"
        "- Prioritize clarity, revenue impact, and implementation practicality.\n"
        "- Be direct and operator-friendly.\n\n"
    )
    return (
        "Analyze this restaurant menu and return a critique using the required JSON schema.\n\n"
        f"Mode: {mode}\n"
        f"Primary goal: {goal}\n"
        f"Restaurant context: {safe_context}\n\n"
        "Scoring guidance:\n"
        "- clarity: readability, naming, scannability\n"
        "- pricing_psychology: anchors, decoys, price formatting, value framing\n"
        "- upsell_potential: combos, add-ons, sizing, pairings\n"
        "- menu_structure: grouping, flow, hierarchy\n"
        "- dietary_signals: labels for vegetarian/vegan/gluten-free/allergens\n\n"
        f"{mode_specific}"
        "Requirements:\n"
        "- Provide exactly 5 top_5_changes if possible.\n"
        "- Rewrite examples should be concrete menu line upgrades.\n"
        "- In Roast mode, rewrite_examples should preserve the humor in the explanation but keep the rewritten menu line usable.\n"
        "- A/B tests should be realistic for a restaurant menu or online ordering page.\n"
        "- Red flags should call out confusing, risky, or conversion-killing issues.\n"
        "- Keep all output in English.\n\n"
        "Roast calibration (only if mode is Roast): aim for 7/10 funny, 10/10 useful.\n\n"
        "Menu text:\n"
        f"{menu_text}"
    )


def _manual_validate_critique(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")

    for key in ["scores", "top_5_changes", "revenue_levers", "rewrite_examples", "ab_tests", "red_flags"]:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

    scores = data["scores"]
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object.")
    for key in SCORE_KEYS:
        if key not in scores:
            raise ValueError(f"scores missing {key}")
        value = scores[key]
        if not isinstance(value, int):
            raise ValueError(f"scores.{key} must be an integer.")
        if not 0 <= value <= 100:
            raise ValueError(f"scores.{key} must be 0-100.")

    if not isinstance(data["top_5_changes"], list):
        raise ValueError("top_5_changes must be a list.")
    if not all(isinstance(x, str) for x in data["top_5_changes"]):
        raise ValueError("top_5_changes items must be strings.")

    rev = data["revenue_levers"]
    if not isinstance(rev, dict):
        raise ValueError("revenue_levers must be an object.")
    for k in ["conversion", "aov", "margin"]:
        if k not in rev or not isinstance(rev[k], list) or not all(isinstance(x, str) for x in rev[k]):
            raise ValueError(f"revenue_levers.{k} must be a string list.")

    if not isinstance(data["rewrite_examples"], list):
        raise ValueError("rewrite_examples must be a list.")
    for item in data["rewrite_examples"]:
        if not isinstance(item, dict):
            raise ValueError("rewrite_examples items must be objects.")
        for k in ["original", "rewritten", "why_it_helps"]:
            if not isinstance(item.get(k), str):
                raise ValueError(f"rewrite_examples item missing string '{k}'.")

    if not isinstance(data["ab_tests"], list):
        raise ValueError("ab_tests must be a list.")
    for item in data["ab_tests"]:
        if not isinstance(item, dict):
            raise ValueError("ab_tests items must be objects.")
        for k in ["hypothesis", "variant_a", "variant_b", "success_metric"]:
            if not isinstance(item.get(k), str):
                raise ValueError(f"ab_tests item missing string '{k}'.")

    if not isinstance(data["red_flags"], list) or not all(isinstance(x, str) for x in data["red_flags"]):
        raise ValueError("red_flags must be a string list.")

    return data


def analyze_menu_text(
    client: Groq,
    menu_text: str,
    mode: str,
    goal: str,
    context: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    logger.info(
        "Starting menu analysis: mode=%s goal=%s context_provided=%s text_chars=%s model=%s",
        mode,
        goal,
        bool((context or "").strip()),
        len(menu_text or ""),
        TEXT_MODEL,
    )
    system_prompt = _critique_system_prompt()
    user_prompt = _critique_user_prompt(menu_text, mode, goal, context)

    create_kwargs = dict(
        model=TEXT_MODEL,
        temperature=0.35 if mode.lower().startswith("fix") else 1.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_output = ""
    response_format_used = "json_schema"
    usage_summary: dict[str, int | None] | None = None
    try:
        logger.debug("Requesting Groq structured output with json_schema.")
        resp = client.chat.completions.create(
            **create_kwargs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "menu_critic_output",
                    "schema": CRITIQUE_JSON_SCHEMA,
                    "strict": True,
                },
            },
        )
        raw_output = _chat_content_text(resp)
        usage_summary = _usage_from_response(resp)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            logger.warning("Text analysis hit rate limit-like error: %s", exc)
            raise RateLimitLikeError(str(exc)) from exc
        logger.warning("json_schema response format failed (%s). Falling back to json_object.", exc)
        try:
            response_format_used = "json_object"
            resp = client.chat.completions.create(
                **create_kwargs,
                response_format={"type": "json_object"},
            )
            raw_output = _chat_content_text(resp)
            usage_summary = _usage_from_response(resp)
        except Exception as inner_exc:
            if _is_rate_limit_error(inner_exc):
                logger.warning("Fallback text analysis hit rate limit-like error: %s", inner_exc)
                raise RateLimitLikeError(str(inner_exc)) from inner_exc
            logger.exception("Groq text analysis request failed.")
            raise MenuCriticError(f"Groq request failed: {inner_exc}") from inner_exc

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        logger.warning("Text analysis returned invalid JSON. raw_len=%s", len(raw_output))
        raise InvalidJSONResponse(raw_output=raw_output) from exc

    try:
        validated = _manual_validate_critique(parsed)
    except ValueError as exc:
        logger.warning("Text analysis returned JSON with invalid shape: %s", exc)
        raise InvalidJSONResponse(raw_output=raw_output, message=f"JSON shape was invalid: {exc}") from exc

    logger.info(
        "Menu analysis complete: top_changes=%s rewrite_examples=%s ab_tests=%s",
        len(validated.get("top_5_changes", [])),
        len(validated.get("rewrite_examples", [])),
        len(validated.get("ab_tests", [])),
    )
    meta = {
        "model": TEXT_MODEL,
        "response_format": response_format_used,
        "usage": usage_summary or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "raw_output_chars": len(raw_output),
    }
    return validated, raw_output, meta


def dumps_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
