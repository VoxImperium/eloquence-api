import os
import json
import logging
import httpx

logger = logging.getLogger(__name__)

OPENLEGI_MCP_BASE = "https://mcp.openlegi.fr/legifrance/mcp"


def _get_url() -> str:
    token = os.getenv("OPENLEGI_TOKEN", "")
    return f"{OPENLEGI_MCP_BASE}?token={token}"


async def search_jurisprudence(query: str, limit: int = 5) -> list[dict]:
    """Search jurisprudence via OpenLegi MCP tool: rechercher_jurisprudence_judiciaire"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "rechercher_jurisprudence_judiciaire",
            "arguments": {"query": query, "nombre": limit},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(_get_url(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("result", {})
        # The MCP result may be a list directly or wrapped in content
        items = _extract_items(raw)
        results = []
        for item in items:
            results.append({
                "id":          item.get("id", ""),
                "date":        item.get("date", ""),
                "chambre":     item.get("chambre", ""),
                "solution":    item.get("solution", ""),
                "resume":      item.get("resume", item.get("sommaire", "")),
                "numero":      item.get("numero", item.get("numberFull", "")),
                "juridiction": item.get("juridiction", item.get("jurisdiction", "")),
                "source":      "openlegi",
            })
        return results
    except Exception as exc:
        logger.error("OpenLegi search_jurisprudence error for query=%r: %s", query, exc)
        return []


async def search_textes(query: str, limit: int = 5) -> list[dict]:
    """Search legal texts via OpenLegi MCP tool: rechercher_code"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "rechercher_code",
            "arguments": {"query": query, "nombre": limit},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(_get_url(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("result", {})
        items = _extract_items(raw)
        results = []
        for item in items:
            results.append({
                "id":     item.get("id", ""),
                "titre":  item.get("titre", item.get("title", "")),
                "nature": item.get("nature", ""),
                "date":   item.get("date", ""),
                "resume": item.get("resume", item.get("sommaire", "")),
                "url":    item.get("url", ""),
                "source": "openlegi",
            })
        return results
    except Exception as exc:
        logger.error("OpenLegi search_textes error for query=%r: %s", query, exc)
        return []


def _extract_items(raw) -> list:
    """Extract a list of items from a MCP tool result payload."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # MCP may wrap results in content[].text (JSON string) or directly as a list
        content = raw.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        parsed = json.loads(block["text"])
                        if isinstance(parsed, list):
                            return parsed
                        if isinstance(parsed, dict):
                            for key in ("results", "items", "decisions", "articles"):
                                if isinstance(parsed.get(key), list):
                                    return parsed[key]
                    except (json.JSONDecodeError, KeyError):
                        pass
        for key in ("results", "items", "decisions", "articles"):
            if isinstance(raw.get(key), list):
                return raw[key]
    return []
