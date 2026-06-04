# zie_capture.py — fires after every notebook Q&A pair
# Writes to openclaw-saas Railway Postgres: zie_training_records + zie_preference_pairs
#
# Schema: 0003_zie_flywheel.sql + 0005_zie_domain_columns.sql (live flywheel schema)
# NOT the factory schema (0003_zie_factory.sql) — no FK columns, no workspace_id, no tenant_id.
#
# v1 note: open-notebook uses a single model per session.
# fast_response and slow_response will be identical in v1.
# The preference pair is still written so the judge route can score it later.
# When a dual-model path is wired (Dip 1 = local/fast, Dip 2 = remote/slow),
# the preference pair becomes meaningful.

import asyncpg
import hashlib
import json
import os
from loguru import logger

ZIE_DATABASE_URL = os.environ.get("ZIE_DATABASE_URL", "")
ZIE_DOMAIN = "notebook"
ZIE_TASK_TYPE = "notebook_qa"


async def capture_qa_pair(
    question: str,
    fast_response: str,
    slow_response: str,
    source_doc_ids: list[str],
) -> None:
    """
    Called after every notebook Q&A. Fire-and-forget via asyncio.create_task().
    Writes two training records (fast + slow) and one preference pair.

    ON CONFLICT behaviour:
    - zie_training_records: DO UPDATE SET source_kind = EXCLUDED.source_kind
      Forces RETURNING id to always return the row UUID, whether inserted or conflicted.
      (ON CONFLICT DO NOTHING returns NULL on conflict — breaks the fast_id/slow_id gate.)
    - zie_preference_pairs: ON CONFLICT (prompt_hash) DO NOTHING
      Unique index idx_zie_preference_pairs_prompt_hash_unique applied in migration 0008.
    """
    if not ZIE_DATABASE_URL:
        logger.warning("[ZIE] ZIE_DATABASE_URL not set — skipping capture")
        return

    prompt_hash = hashlib.sha256(
        json.dumps({"q": question, "docs": sorted(source_doc_ids)}).encode()
    ).hexdigest()

    conn = await asyncpg.connect(ZIE_DATABASE_URL)
    try:
        # ── Fast (cheap / direct_call) training record ────────────────────────
        # DO UPDATE SET source_kind = EXCLUDED.source_kind is a no-op update that
        # forces RETURNING id to return the existing row UUID on conflict.
        fast_id = await conn.fetchval(
            """
            INSERT INTO zie_training_records
              (task_type, domain, prompt_hash, prompt_json,
               remote_response_json, quality_score, source_kind)
            VALUES ($1, $2, $3, $4, $5, 0.5, 'direct_call')
            ON CONFLICT (prompt_hash)
              DO UPDATE SET source_kind = EXCLUDED.source_kind
            RETURNING id
            """,
            ZIE_TASK_TYPE,
            ZIE_DOMAIN,
            prompt_hash + "_fast",
            json.dumps({"question": question, "source_doc_ids": source_doc_ids}),
            json.dumps({"answer": fast_response}),
        )

        # ── Slow (quality / remote_promoted) training record ─────────────────
        slow_id = await conn.fetchval(
            """
            INSERT INTO zie_training_records
              (task_type, domain, prompt_hash, prompt_json,
               remote_response_json, quality_score, source_kind)
            VALUES ($1, $2, $3, $4, $5, 1.0, 'remote_promoted')
            ON CONFLICT (prompt_hash)
              DO UPDATE SET source_kind = EXCLUDED.source_kind
            RETURNING id
            """,
            ZIE_TASK_TYPE,
            ZIE_DOMAIN,
            prompt_hash + "_slow",
            json.dumps({"question": question, "source_doc_ids": source_doc_ids}),
            json.dumps({"answer": slow_response}),
        )

        # ── Preference pair ───────────────────────────────────────────────────
        # chosen = slow/quality response, rejected = fast/cheap response
        # Unique constraint on prompt_hash (migration 0008) — skip on duplicate.
        if fast_id and slow_id:
            try:
                await conn.execute(
                    """
                    INSERT INTO zie_preference_pairs
                      (task_type, domain, prompt_hash,
                       chosen_response_json, rejected_response_json,
                       preference_source, source_kind)
                    VALUES ($1, $2, $3, $4, $5, 'remote_beats_local', 'direct_call')
                    ON CONFLICT (prompt_hash) DO NOTHING
                    """,
                    ZIE_TASK_TYPE,
                    ZIE_DOMAIN,
                    prompt_hash,
                    json.dumps({"answer": slow_response}),
                    json.dumps({"answer": fast_response}),
                )
                logger.debug(
                    f"[ZIE] captured notebook_qa | hash={prompt_hash[:12]}... "
                    f"fast_id={fast_id} slow_id={slow_id}"
                )
            except asyncpg.UniqueViolationError:
                logger.debug(f"[ZIE] preference pair already exists for hash={prompt_hash[:12]}...")
        else:
            logger.warning(f"[ZIE] one or both training record IDs null — skipping preference pair")

    except Exception as e:
        logger.error(f"[ZIE] capture_qa_pair failed: {e}")
    finally:
        await conn.close()
