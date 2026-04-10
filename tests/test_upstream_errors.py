"""
Unit tests for upstream error handling (v1.6.9).

Tests cover:
  - OCI content filter (400 "Inappropriate content detected") → 400 + content_filter_error
  - OCI empty response / silent rate-limiting (500 "OCI model returned an empty response") → 503
  - No retry on 4xx (content filter errors must not be retried)
  - INFO log contains last user message on content filter
"""
import json
import logging
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

import app.main as main_module
from app.main import app
from app.services.vllm_client import VLLMClient
from tests.conftest import DEFAULT_TOOLS, SYSTEM_MSG, user_msg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_http_status_error(status_code: int, body: str) -> httpx.HTTPStatusError:
    """Build a realistic httpx.HTTPStatusError with the given status and body."""
    request = httpx.Request("POST", "http://oci-proxy:8005/v1/chat/completions")
    response = httpx.Response(status_code, text=body, request=request)
    return httpx.HTTPStatusError(
        f"{'Client' if status_code < 500 else 'Server'} error '{status_code}' for url '...'",
        request=request,
        response=response,
    )


def _post(client, messages=None, tools=None):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": messages or [SYSTEM_MSG, user_msg("Hello")],
            "tools": tools if tools is not None else DEFAULT_TOOLS,
        },
    )


# ── Content Filter (400) ──────────────────────────────────────────────────────

class TestContentFilterError:

    def test_content_filter_returns_400(self, client):
        """400 from OCI with 'Inappropriate content detected' → toolproxy returns 400."""
        err = _make_http_status_error(400, '{"detail":"Inappropriate content detected!!!"}')
        mock = AsyncMock(side_effect=err)
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            resp = _post(client)
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "content_filter_error"
        assert "content filter" in body["error"]["message"].lower()

    def test_content_filter_logs_last_user_message(self, client, caplog):
        """On content filter, the last user message must appear in the INFO log."""
        err = _make_http_status_error(400, '{"detail":"Inappropriate content detected!!!"}')
        mock = AsyncMock(side_effect=err)
        trigger_text = "this is the suspicious user message"
        messages = [SYSTEM_MSG, user_msg(trigger_text)]
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            with caplog.at_level(logging.INFO, logger="app.main"):
                _post(client, messages=messages)
        log_text = caplog.text
        assert trigger_text in log_text, (
            f"Expected last user message to appear in INFO log. Log was:\n{log_text}"
        )

    def test_content_filter_logs_last_user_message_not_system(self, client, caplog):
        """The logged message must be the user turn, not the system prompt."""
        err = _make_http_status_error(400, '{"detail":"Inappropriate content detected!!!"}')
        mock = AsyncMock(side_effect=err)
        system_text = "you are a helpful assistant"
        user_text = "help me with this task"
        messages = [
            {"role": "system", "content": system_text},
            user_msg(user_text),
        ]
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            with caplog.at_level(logging.INFO, logger="app.main"):
                _post(client, messages=messages)
        # user turn must be logged
        assert user_text in caplog.text
        # system content should NOT be in the content-filter INFO line
        info_lines = [l for l in caplog.text.splitlines() if "Content-filter trigger" in l]
        assert info_lines, "Expected 'Content-filter trigger' line in logs"
        assert system_text not in info_lines[0]

    def test_other_400_not_treated_as_content_filter(self, client):
        """A generic 400 (not content filter) must still return 502 (generic upstream error)."""
        err = _make_http_status_error(400, '{"detail":"Bad request — invalid parameter"}')
        mock = AsyncMock(side_effect=err)
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            resp = _post(client)
        # Not a content filter error — must be a generic 502
        assert resp.status_code == 502
        assert resp.json()["error"]["type"] == "upstream_error"


# ── Silent Rate-Limiting / Empty Response (500) ───────────────────────────────

class TestEmptyResponseError:

    def test_empty_response_returns_503(self, client):
        """500 from oci-proxy with 'OCI model returned an empty response' → 503."""
        err = _make_http_status_error(500, '{"detail":"OCI model returned an empty response."}')
        mock = AsyncMock(side_effect=err)
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            resp = _post(client)
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["type"] == "rate_limit_error"
        assert "transient" in body["error"]["message"].lower()

    def test_empty_response_error_message_mentions_retry(self, client):
        """The 503 error message must tell the user to retry."""
        err = _make_http_status_error(500, '{"detail":"OCI model returned an empty response."}')
        mock = AsyncMock(side_effect=err)
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            resp = _post(client)
        assert "retry" in resp.json()["error"]["message"].lower()

    def test_other_500_not_treated_as_empty_response(self, client):
        """A generic 500 must still return 502, not 503."""
        err = _make_http_status_error(500, '{"detail":"Internal server error"}')
        mock = AsyncMock(side_effect=err)
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            resp = _post(client)
        assert resp.status_code == 502
        assert resp.json()["error"]["type"] == "upstream_error"


# ── No retry on 4xx ───────────────────────────────────────────────────────────

class TestNoRetryOn4xx:
    """
    vllm_client must NOT retry 4xx errors — they are client errors that
    won't improve with retries (content filter, bad request, etc.).
    """

    def test_4xx_raises_immediately_without_retry(self):
        """A 400 error must be raised on the first attempt, no retry loop."""
        import asyncio
        call_count = 0

        async def _run():
            nonlocal call_count

            async def _mock_post(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                request = httpx.Request("POST", "http://upstream/v1/chat/completions")
                response = httpx.Response(400, text='{"detail":"Inappropriate content detected!!!"}', request=request)
                raise httpx.HTTPStatusError("Client error '400'", request=request, response=response)

            uc = VLLMClient(base_url="http://upstream/v1", model="test", max_retries=3)
            uc.client.post = _mock_post  # type: ignore
            with pytest.raises(httpx.HTTPStatusError):
                await uc.chat_completion(messages=[{"role": "user", "content": "hi"}])

        asyncio.run(_run())
        assert call_count == 1, (
            f"4xx error was retried {call_count} times — expected exactly 1 attempt. "
            "4xx errors must never be retried."
        )

    def test_5xx_is_retried(self):
        """A 500 error must still be retried (transient upstream errors)."""
        import asyncio
        call_count = 0

        async def _run():
            nonlocal call_count

            async def _mock_post(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                request = httpx.Request("POST", "http://upstream/v1/chat/completions")
                response = httpx.Response(500, text='{"detail":"Internal server error"}', request=request)
                raise httpx.HTTPStatusError("Server error '500'", request=request, response=response)

            uc = VLLMClient(base_url="http://upstream/v1", model="test", max_retries=2)
            uc.client.post = _mock_post  # type: ignore
            with pytest.raises(httpx.HTTPStatusError):
                await uc.chat_completion(messages=[{"role": "user", "content": "hi"}])

        asyncio.run(_run())
        assert call_count == 2, (
            f"Expected 5xx to be retried once (2 total attempts), got {call_count} attempts."
        )
