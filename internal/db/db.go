package db

import (
	"database/sql"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

//go:embed schema.sql
var schemaSQL string

type DB struct {
	conn *sql.DB
}

// Task represents a unit of work in the agent hierarchy.
type Task struct {
	ID        int64
	ParentID  *int64
	Goal      string
	Status    string
	Result    *string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Message represents a single conversation turn stored in agent_state.
type Message struct {
	ID         int64
	TaskID     int64
	Step       int
	Role       string
	Content    string
	ToolCalls  *string // JSON-encoded tool calls from assistant
	ToolCallID *string // ID linking tool result back to its call
	CreatedAt  time.Time
}

// Result stores structured research output.
type Result struct {
	ID        int64
	TaskID    int64
	Kind      string
	Data      json.RawMessage
	CreatedAt time.Time
}

func Open(path string) (*DB, error) {
	conn, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_foreign_keys=ON&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	conn.SetMaxOpenConns(1) // SQLite handles one writer at a time
	if err := migrate(conn); err != nil {
		conn.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &DB{conn: conn}, nil
}

func (d *DB) Close() error {
	return d.conn.Close()
}

func migrate(conn *sql.DB) error {
	_, err := conn.Exec(schemaSQL)
	return err
}

// CreateTask inserts a new task and returns its ID.
func (d *DB) CreateTask(parentID *int64, goal string) (int64, error) {
	res, err := d.conn.Exec(
		"INSERT INTO tasks (parent_id, goal, status) VALUES (?, ?, 'pending')",
		parentID, goal,
	)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// UpdateTaskStatus sets the status (and optionally result) of a task.
func (d *DB) UpdateTaskStatus(id int64, status string, result *string) error {
	_, err := d.conn.Exec(
		"UPDATE tasks SET status = ?, result = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
		status, result, id,
	)
	return err
}

// GetTask fetches a single task by ID.
func (d *DB) GetTask(id int64) (*Task, error) {
	row := d.conn.QueryRow("SELECT id, parent_id, goal, status, result, created_at, updated_at FROM tasks WHERE id = ?", id)
	return scanTask(row)
}

// GetLatestRootTask returns the most recent root task that is not done.
func (d *DB) GetLatestRootTask() (*Task, error) {
	row := d.conn.QueryRow(
		"SELECT id, parent_id, goal, status, result, created_at, updated_at FROM tasks WHERE parent_id IS NULL AND status != 'done' ORDER BY id DESC LIMIT 1",
	)
	return scanTask(row)
}

// GetChildTasks returns all sub-tasks of a given parent.
func (d *DB) GetChildTasks(parentID int64) ([]Task, error) {
	rows, err := d.conn.Query(
		"SELECT id, parent_id, goal, status, result, created_at, updated_at FROM tasks WHERE parent_id = ? ORDER BY id",
		parentID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var tasks []Task
	for rows.Next() {
		t, err := scanTaskRow(rows)
		if err != nil {
			return nil, err
		}
		tasks = append(tasks, t)
	}
	return tasks, rows.Err()
}

// CountRunningChildren returns the number of running sub-tasks for a parent.
func (d *DB) CountRunningChildren(parentID int64) (int, error) {
	var count int
	err := d.conn.QueryRow(
		"SELECT COUNT(*) FROM tasks WHERE parent_id = ? AND status = 'running'",
		parentID,
	).Scan(&count)
	return count, err
}

// AppendMessage adds a conversation message to agent_state.
func (d *DB) AppendMessage(taskID int64, step int, role, content string, toolCalls, toolCallID *string) (int64, error) {
	res, err := d.conn.Exec(
		"INSERT INTO agent_state (task_id, step, role, content, tool_calls, tool_call_id) VALUES (?, ?, ?, ?, ?, ?)",
		taskID, step, role, content, toolCalls, toolCallID,
	)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// GetMessages returns all messages for a task, ordered by step then id.
func (d *DB) GetMessages(taskID int64) ([]Message, error) {
	rows, err := d.conn.Query(
		"SELECT id, task_id, step, role, content, tool_calls, tool_call_id, created_at FROM agent_state WHERE task_id = ? ORDER BY step, id",
		taskID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var msgs []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.TaskID, &m.Step, &m.Role, &m.Content, &m.ToolCalls, &m.ToolCallID, &m.CreatedAt); err != nil {
			return nil, err
		}
		msgs = append(msgs, m)
	}
	return msgs, rows.Err()
}

// GetMessageCount returns the number of messages for a task.
func (d *DB) GetMessageCount(taskID int64) (int, error) {
	var count int
	err := d.conn.QueryRow("SELECT COUNT(*) FROM agent_state WHERE task_id = ?", taskID).Scan(&count)
	return count, err
}

// DeleteMessagesByIDs removes messages by their IDs (used during context compaction).
func (d *DB) DeleteMessagesByIDs(ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, id := range ids {
		if _, err := tx.Exec("DELETE FROM agent_state WHERE id = ?", id); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// SaveResult stores a structured research finding.
func (d *DB) SaveResult(taskID int64, kind string, data json.RawMessage) (int64, error) {
	res, err := d.conn.Exec(
		"INSERT INTO results (task_id, kind, data) VALUES (?, ?, ?)",
		taskID, kind, string(data),
	)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// GetResults returns all results for a task.
func (d *DB) GetResults(taskID int64) ([]Result, error) {
	rows, err := d.conn.Query(
		"SELECT id, task_id, kind, data, created_at FROM results WHERE task_id = ? ORDER BY id",
		taskID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var results []Result
	for rows.Next() {
		var r Result
		var data string
		if err := rows.Scan(&r.ID, &r.TaskID, &r.Kind, &data, &r.CreatedAt); err != nil {
			return nil, err
		}
		r.Data = json.RawMessage(data)
		results = append(results, r)
	}
	return results, rows.Err()
}

// FindDuplicateResult checks if a result with the same kind exists where the
// JSON data contains matching name and company (case-insensitive, normalized).
func (d *DB) FindDuplicateResult(taskID int64, kind, normName, normCompany string) (*Result, error) {
	rows, err := d.conn.Query(
		"SELECT id, task_id, kind, data, created_at FROM results WHERE task_id = ? AND kind = ?",
		taskID, kind,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var r Result
		var data string
		if err := rows.Scan(&r.ID, &r.TaskID, &r.Kind, &data, &r.CreatedAt); err != nil {
			return nil, err
		}
		r.Data = json.RawMessage(data)

		// Extract name and company from stored data for comparison
		var stored struct {
			Name    string `json:"name"`
			Company string `json:"company"`
		}
		json.Unmarshal(r.Data, &stored)

		storedName := strings.ToLower(strings.TrimSpace(stored.Name))
		storedCompany := strings.ToLower(strings.TrimSpace(stored.Company))

		// Remove parenthetical from stored name too
		if idx := strings.Index(storedName, "("); idx > 0 {
			storedName = strings.TrimSpace(storedName[:idx])
		}

		// Match if either name contains the other (handles "Pat Opet" vs "Patrick Opet")
		nameMatch := strings.Contains(storedName, normName) || strings.Contains(normName, storedName)
		// Match if company names overlap after normalization
		companyMatch := strings.Contains(storedCompany, normCompany) || strings.Contains(normCompany, storedCompany)

		if nameMatch && companyMatch {
			return &r, nil
		}
	}
	return nil, rows.Err()
}

// UpdateResult updates the data field of an existing result.
func (d *DB) UpdateResult(id int64, data json.RawMessage) error {
	_, err := d.conn.Exec("UPDATE results SET data = ? WHERE id = ?", string(data), id)
	return err
}

// CountUniqueResults returns the count of results for a task and kind.
func (d *DB) CountUniqueResults(taskID int64, kind string) (int, error) {
	var count int
	err := d.conn.QueryRow("SELECT COUNT(*) FROM results WHERE task_id = ? AND kind = ?", taskID, kind).Scan(&count)
	return count, err
}

// SetKV sets a key-value pair.
func (d *DB) SetKV(key, value string) error {
	_, err := d.conn.Exec(
		"INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
		key, value,
	)
	return err
}

// GetKV retrieves a value by key. Returns empty string if not found.
func (d *DB) GetKV(key string) (string, error) {
	var value string
	err := d.conn.QueryRow("SELECT value FROM kv WHERE key = ?", key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return value, err
}

type scannable interface {
	Scan(dest ...any) error
}

func scanTask(row scannable) (*Task, error) {
	var t Task
	err := row.Scan(&t.ID, &t.ParentID, &t.Goal, &t.Status, &t.Result, &t.CreatedAt, &t.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &t, nil
}

func scanTaskRow(rows *sql.Rows) (Task, error) {
	var t Task
	err := rows.Scan(&t.ID, &t.ParentID, &t.Goal, &t.Status, &t.Result, &t.CreatedAt, &t.UpdatedAt)
	return t, err
}
