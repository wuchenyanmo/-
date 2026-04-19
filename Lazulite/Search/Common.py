from __future__ import annotations

import re
import unicodedata

from fuzzywuzzy import fuzz

RE_SEARCH_BRACKETS = re.compile(r"[\(\[（【].*?[\)\]）】]")
RE_SEARCH_PUNCT = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\s'-]+")
RE_SEARCH_SPACES = re.compile(r"\s+")
RE_DASH_VARIANTS = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")


def combined_fuzzy_score(str1: str, str2: str, full_match_weight: float = 0.2) -> float:
    partial = fuzz.partial_ratio(str1, str2)
    full = fuzz.ratio(str1, str2)
    return float(partial * (1 - full_match_weight) + full * full_match_weight)


def normalize_search_text(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not value:
        return ""
    value = RE_DASH_VARIANTS.sub("-", value)
    value = value.replace("☆", " ").replace("★", " ").replace("～", " ").replace("~", " ")
    value = RE_SEARCH_SPACES.sub(" ", value).strip()
    return value


def strip_search_bracket_suffix(text: str | None) -> str:
    value = normalize_search_text(text)
    if not value:
        return ""
    value = RE_SEARCH_BRACKETS.sub(" ", value)
    value = RE_SEARCH_SPACES.sub(" ", value).strip(" ._-")
    return value


def simplify_search_text(text: str | None) -> str:
    value = strip_search_bracket_suffix(text)
    if not value:
        return ""
    value = RE_SEARCH_PUNCT.sub(" ", value)
    value = RE_SEARCH_SPACES.sub(" ", value).strip(" ._-")
    return value


def build_search_text_variants(text: str | None) -> list[str]:
    variants: list[str] = []
    for candidate in [
        str(text or "").strip(),
        normalize_search_text(text),
        strip_search_bracket_suffix(text),
        simplify_search_text(text),
    ]:
        value = RE_SEARCH_SPACES.sub(" ", candidate).strip()
        if value and value not in variants:
            variants.append(value)
    return variants
