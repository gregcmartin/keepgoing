package skills

import (
	"context"
	"encoding/json"
	"fmt"

	"keepgoing/internal/db"
)

// SaveFindingSkill lets the agent persist structured research results to the database.
type SaveFindingSkill struct {
	DB     *db.DB
	TaskID int64
}

type saveFindingParams struct {
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data"`
}

func (s *SaveFindingSkill) Name() string { return "save_finding" }

func (s *SaveFindingSkill) Description() string {
	return "Save a structured research finding to the database. Use this to record companies, CISOs, contact info, or any other research output. The data field should be a JSON object with relevant fields."
}

func (s *SaveFindingSkill) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"kind": {
				"type": "string",
				"description": "The type of finding: company, ciso, contact, note"
			},
			"data": {
				"type": "object",
				"description": "The structured data for this finding (any JSON object)"
			}
		},
		"required": ["kind", "data"]
	}`)
}

func (s *SaveFindingSkill) Execute(ctx context.Context, params json.RawMessage) (string, error) {
	var p saveFindingParams
	if err := json.Unmarshal(params, &p); err != nil {
		return "", fmt.Errorf("parse params: %w", err)
	}
	id, err := s.DB.SaveResult(s.TaskID, p.Kind, p.Data)
	if err != nil {
		return "", fmt.Errorf("save result: %w", err)
	}
	return fmt.Sprintf("Finding saved (id=%d, kind=%s)", id, p.Kind), nil
}
