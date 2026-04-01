PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id   INTEGER REFERENCES tasks(id),
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','done','failed')),
    result      TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_state (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES tasks(id),
    step        INTEGER NOT NULL DEFAULT 0,
    role        TEXT NOT NULL CHECK(role IN ('system','user','assistant','tool')),
    content     TEXT NOT NULL,
    tool_calls  TEXT,
    tool_call_id TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES tasks(id),
    kind        TEXT NOT NULL,
    data        TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
