"""Rate limiting, 429 quota exception handling, and backoff utilities for AURA.AI.

Inspects Google Gen AI ClientError responses, extracts official retry delays,
distinguishes per-minute temporary rate limits from daily quota exhaustion,
and ensures requests are not repeatedly retried when daily quotas are spent.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from google.genai import errors

from utils.helpers import sanitize_error_message
from utils.usage_tracker import record_api_request


@dataclass
class QuotaErrorDetails:
    """Structured analysis of a Gemini API quota / rate-limit response."""

    is_429: bool
    is_daily_limit: bool
    retry_delay_seconds: float | None
    model_name: str
    raw_message: str
    friendly_message: str


def parse_quota_error(exc: Exception, model_name: str = "") -> QuotaErrorDetails:
    """Parse an exception to detect HTTP 429 (RESOURCE_EXHAUSTED), extract retry_delay,
    and classify between temporary per-minute rate limits and daily quota exhaustion.
    """
    raw_str = str(exc)
    err_lower = raw_str.lower()

    # 1. Determine strictly if this is an authentic 429 / RESOURCE_EXHAUSTED error
    has_code_429 = getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429
    is_code_429 = (isinstance(exc, errors.ClientError) and getattr(exc, "code", None) == 429) or has_code_429
    is_canonical_429 = (
        "resource_exhausted" in err_lower
        or "resource has been exhausted" in err_lower
        or "resource exhausted" in err_lower
        or "http 429" in err_lower
        or "status: 429" in err_lower
        or "status_code: 429" in err_lower
        or "code: 429" in err_lower
        or "code 429" in err_lower
        or "429 too many requests" in err_lower
    )
    is_429 = is_code_429 or is_canonical_429

    if not is_429:
        return QuotaErrorDetails(
            is_429=False,
            is_daily_limit=False,
            retry_delay_seconds=None,
            model_name=model_name,
            raw_message=raw_str,
            friendly_message=sanitize_error_message(raw_str),
        )

    # 2. Check for daily quota exhaustion vs temporary per-minute rate limit
    daily_indicators = [
        "generate_content_free_tier_requests",
        "free_tier_requests",
        "free tier requests",
        "requests per day",
        "requests_per_day",
        "daily quota",
        "daily limit",
        "daily_quota",
        "daily_limit",
        "per day",
        "per_day",
        "generatecontentrequestsperday",
        "generatecontent requests per day",
        "generate_content_requests_per_day",
        "free_tier_requests_per_day",
        "day quota",
        "day_quota",
    ]
    err_normalized = err_lower.replace(" ", "_")
    is_daily_limit = any(ind in err_lower for ind in daily_indicators) or any(
        ind in err_normalized for ind in daily_indicators
    )

    # 3. Parse Google's retry_delay from headers, details, or error message
    retry_delay: float | None = None

    # Check response headers if present on exception
    response_obj = getattr(exc, "response", None)
    if response_obj and hasattr(response_obj, "headers"):
        headers = response_obj.headers or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                retry_delay = float(retry_after)
            except (ValueError, TypeError):
                pass

    # Check RPC details array (type.googleapis.com/google.rpc.RetryInfo)
    details = getattr(exc, "details", None)
    if details and isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                # Search for retryDelay or retry_delay
                rd = item.get("retryDelay") or item.get("retry_delay")
                if rd:
                    if isinstance(rd, (int, float)):
                        retry_delay = float(rd)
                        break
                    if isinstance(rd, str):
                        m = re.search(r"([\d\.]+)\s*s?", rd)
                        if m:
                            try:
                                retry_delay = float(m.group(1))
                                break
                            except ValueError:
                                pass
                # Search for seconds in dictionary
                if isinstance(rd, dict) and "seconds" in rd:
                    try:
                        retry_delay = float(rd["seconds"])
                        break
                    except (ValueError, TypeError):
                        pass

    # Regex search in raw error text
    if retry_delay is None:
        patterns = [
            r"retry after\s+([\d\.]+)\s*s?",
            r"retry_delay\s*[:=]\s*['\"]?([\d\.]+)s?['\"]?",
            r"quotaresetdelay\s*[:=]\s*['\"]?([\d\.]+)s?['\"]?",
            r"wait\s+([\d\.]+)\s*s(?:econds?)?",
            r"in\s+([\d\.]+)\s*s(?:econds?)?",
            r"reset in\s+([\d\.]+)\s*s",
            r"retry in\s+([\d\.]+)\s*s",
            r"([\d\.]+)\s*s\s*(?:quota\s*reset|wait)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_str, re.IGNORECASE)
            if match:
                try:
                    retry_delay = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass

    # 4. Formulate clear, user-friendly message
    clean_model = model_name.replace("models/", "") if model_name else "Gemini"
    if is_daily_limit:
        friendly = (
            f"Google Gemini daily request quota has been exhausted for model '{clean_model}'. "
            f"Google resets daily free-tier quotas at midnight Pacific Time (00:00 UTC-8). "
            f"Please wait for quota reset, configure a different API key, or switch to a paid tier."
        )
    elif retry_delay is not None:
        wait_int = max(1, int(round(retry_delay)))
        friendly = (
            f"Google Gemini temporary rate limit reached (HTTP 429). "
            f"Estimated retry wait: ~{wait_int} second{'s' if wait_int != 1 else ''}. "
            f"Please wait a moment before sending another prompt."
        )
    else:
        friendly = (
            f"Google Gemini temporary rate limit reached (HTTP 429: RESOURCE_EXHAUSTED). "
            f"Request volume exceeded the per-minute threshold. Please wait a few moments and try again."
        )

    return QuotaErrorDetails(
        is_429=True,
        is_daily_limit=is_daily_limit,
        retry_delay_seconds=retry_delay,
        model_name=model_name,
        raw_message=raw_str,
        friendly_message=friendly,
    )


def execute_with_quota_guard(
    fn: Callable[[], Any],
    request_type: str,
    model_name: str,
    max_retries: int = 2,
    initial_backoff: float = 1.5,
) -> Any:
    """Execute an API function wrapped in quota-aware retry logic.

    Rules:
    - On success: records successful telemetry and returns result.
    - If 429 with daily quota exhaustion: DOES NOT RETRY; raises immediately.
    - If 429 with temporary rate limit:
        - If retry_delay is <= 6.0s and attempts remain: sleeps and retries.
        - If retry_delay > 6.0s or retries exhausted: raises immediately with user-friendly retry wait time.
    - If non-429 error: records error and raises immediately.
    """
    backoff = initial_backoff
    last_quota_error: QuotaErrorDetails | None = None
    reservation_id: int | None = None

    if request_type != "verification":
        from utils.usage_tracker import reserve_api_request
        reservation_id = reserve_api_request(request_type=request_type, model_name=model_name)

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            result = fn()
            latency = time.time() - t0
            record_api_request(
                request_type=request_type,
                status="success",
                latency=latency,
                model_name=model_name,
                reservation_id=reservation_id,
            )
            return result
        except errors.ClientError as exc:
            latency = time.time() - t0
            quota_info = parse_quota_error(exc, model_name=model_name)
            last_quota_error = quota_info

            if quota_info.is_429:
                record_api_request(
                    request_type=request_type,
                    status="429_quota_exceeded",
                    latency=latency,
                    is_429=True,
                    error_message=quota_info.friendly_message,
                    model_name=model_name,
                    reservation_id=reservation_id,
                )
                reservation_id = None

                # Never retry if daily quota is spent
                if quota_info.is_daily_limit:
                    raise RuntimeError(quota_info.friendly_message) from None

                # For temporary rate limits, check if delay is reasonable to wait
                delay = quota_info.retry_delay_seconds
                if delay is not None and delay <= 6.0 and attempt < max_retries:
                    time.sleep(delay)
                    continue
                elif delay is None and attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    raise RuntimeError(quota_info.friendly_message) from None
            else:
                record_api_request(
                    request_type=request_type,
                    status="error",
                    latency=latency,
                    error_message=str(exc),
                    model_name=model_name,
                    reservation_id=reservation_id,
                )
                reservation_id = None
                raise RuntimeError(sanitize_error_message(str(exc))) from None
        except Exception as exc:
            latency = time.time() - t0
            quota_info = parse_quota_error(exc, model_name=model_name)
            if quota_info.is_429:
                record_api_request(
                    request_type=request_type,
                    status="429_quota_exceeded",
                    latency=latency,
                    is_429=True,
                    error_message=quota_info.friendly_message,
                    model_name=model_name,
                    reservation_id=reservation_id,
                )
                reservation_id = None
                if quota_info.is_daily_limit or attempt >= max_retries:
                    raise RuntimeError(quota_info.friendly_message) from None
                time.sleep(backoff)
                backoff *= 2.0
                continue

            record_api_request(
                request_type=request_type,
                status="error",
                latency=latency,
                error_message=str(exc),
                model_name=model_name,
                reservation_id=reservation_id,
            )
            reservation_id = None
            raise RuntimeError(sanitize_error_message(str(exc))) from None

    if last_quota_error:
        raise RuntimeError(last_quota_error.friendly_message)
    raise RuntimeError("API request failed after retries.")
