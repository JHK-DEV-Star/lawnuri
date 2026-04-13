-- ==========================================================================
-- LawNuri database schema (SQLite-compatible)
--
-- All tables use IF NOT EXISTS so the schema can be applied idempotently
-- on every application startup.
-- ==========================================================================

-- --------------------------------------------------------------------------
-- Application settings (singleton row)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    data        TEXT    NOT NULL DEFAULT '{}',
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- --------------------------------------------------------------------------
-- Debate sessions
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS debates (
    debate_id       TEXT    PRIMARY KEY,
    situation_brief TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'created',
    analysis        TEXT    DEFAULT NULL,
    current_round   INTEGER NOT NULL DEFAULT 0,
    state           TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_debates_status     ON debates (status);
CREATE INDEX IF NOT EXISTS idx_debates_created_at ON debates (created_at);

-- --------------------------------------------------------------------------
-- Uploaded documents (per debate)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uploaded_documents (
    doc_id      TEXT    PRIMARY KEY,
    debate_id   TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    content     TEXT    NOT NULL DEFAULT '',
    mime_type   TEXT    NOT NULL DEFAULT 'text/plain',
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (debate_id) REFERENCES debates (debate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_uploaded_documents_debate
    ON uploaded_documents (debate_id);

-- --------------------------------------------------------------------------
-- Vector chunks (embeddings stored as BLOBs)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector_chunks (
    chunk_id    TEXT    PRIMARY KEY,
    debate_id   TEXT    NOT NULL,
    pool        TEXT    NOT NULL DEFAULT 'default',
    content     TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (debate_id) REFERENCES debates (debate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_debate_pool
    ON vector_chunks (debate_id, pool);

-- --------------------------------------------------------------------------
-- Knowledge graph entities
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id   TEXT    PRIMARY KEY,
    debate_id   TEXT    NOT NULL,
    pool        TEXT    NOT NULL DEFAULT 'default',
    label       TEXT    NOT NULL,
    entity_type TEXT    NOT NULL,
    properties  TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (debate_id) REFERENCES debates (debate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_entities_debate_pool
    ON graph_entities (debate_id, pool);
CREATE INDEX IF NOT EXISTS idx_graph_entities_type
    ON graph_entities (entity_type);

-- --------------------------------------------------------------------------
-- Knowledge graph relations
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_relations (
    relation_id     TEXT    PRIMARY KEY,
    debate_id       TEXT    NOT NULL,
    pool            TEXT    NOT NULL DEFAULT 'default',
    source_id       TEXT    NOT NULL,
    target_id       TEXT    NOT NULL,
    relation_type   TEXT    NOT NULL,
    properties      TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (debate_id)  REFERENCES debates (debate_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id)  REFERENCES graph_entities (entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id)  REFERENCES graph_entities (entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_relations_debate_pool
    ON graph_relations (debate_id, pool);
CREATE INDEX IF NOT EXISTS idx_graph_relations_source
    ON graph_relations (source_id);
CREATE INDEX IF NOT EXISTS idx_graph_relations_target
    ON graph_relations (target_id);

-- --------------------------------------------------------------------------
-- Legal API response cache
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS legal_api_cache (
    cache_key   TEXT    PRIMARY KEY,
    debate_id   TEXT    NOT NULL,
    data        TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at  TEXT    DEFAULT NULL,
    FOREIGN KEY (debate_id) REFERENCES debates (debate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_legal_api_cache_debate
    ON legal_api_cache (debate_id);
CREATE INDEX IF NOT EXISTS idx_legal_api_cache_expires
    ON legal_api_cache (expires_at);

-- --------------------------------------------------------------------------
-- PII anonymization mappings (per debate)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anonymization_maps (
    debate_id   TEXT    PRIMARY KEY,
    mapping     TEXT    NOT NULL DEFAULT '{}',
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (debate_id) REFERENCES debates (debate_id) ON DELETE CASCADE
);

-- --------------------------------------------------------------------------
-- Evidence debug log — tracks how each evidence ID was generated
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_debug_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id   TEXT,
    round       INTEGER,
    speaker     TEXT,
    result_keys TEXT,
    case_number TEXT,
    law_name    TEXT,
    title       TEXT,
    source      TEXT,
    real_id     TEXT,
    is_uuid     INTEGER DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
