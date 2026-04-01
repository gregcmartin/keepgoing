package skills

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"keepgoing/internal/db"
)

// SaveFindingSkill lets the agent persist structured research results to the database.
// Includes deduplication: if a CISO with the same normalized name+company already exists,
// the existing record is updated (merged) instead of creating a duplicate.
type SaveFindingSkill struct {
	DB     *db.DB
	TaskID int64
}

type saveFindingParams struct {
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data"`
}

// findingData is a loose struct for extracting dedup keys from the JSON data.
type findingData struct {
	Name    string `json:"name"`
	Company string `json:"company"`
	Title   string `json:"title"`
}

func (s *SaveFindingSkill) Name() string { return "save_finding" }

func (s *SaveFindingSkill) Description() string {
	return "Save a structured research finding to the database. Automatically deduplicates — if a CISO with the same name and company already exists, the record is updated with any new information instead of creating a duplicate."
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

	// For CISOs, check for duplicates before saving
	if p.Kind == "ciso" {
		var fd findingData
		json.Unmarshal(p.Data, &fd)

		if fd.Name != "" && fd.Company != "" {
			existing, err := s.DB.FindDuplicateResult(s.TaskID, "ciso", normalizeName(fd.Name), normalizeCompany(fd.Company))
			if err != nil {
				log.Printf("[dedup] Error checking for duplicates: %v", err)
			}
			if existing != nil {
				// Merge new data into existing record
				merged := mergeJSON(existing.Data, p.Data)
				if err := s.DB.UpdateResult(existing.ID, merged); err != nil {
					return "", fmt.Errorf("update existing: %w", err)
				}
				return fmt.Sprintf("DUPLICATE — updated existing CISO record (id=%d, %s at %s). Not counted as new.", existing.ID, fd.Name, fd.Company), nil
			}
		}
	}

	id, err := s.DB.SaveResult(s.TaskID, p.Kind, p.Data)
	if err != nil {
		return "", fmt.Errorf("save result: %w", err)
	}

	// Report unique count so the agent knows its progress
	count, _ := s.DB.CountUniqueResults(s.TaskID, "ciso")
	return fmt.Sprintf("NEW CISO saved (id=%d, %s). Total unique CISOs: %d", id, p.Kind, count), nil
}

// normalizeName strips parenthetical aliases, trims whitespace, lowercases.
// "Pat Opet (William Patrick Opet)" → "pat opet"
func normalizeName(name string) string {
	// Remove anything in parentheses
	if idx := strings.Index(name, "("); idx > 0 {
		name = name[:idx]
	}
	return strings.ToLower(strings.TrimSpace(name))
}

// normalizeCompany strips common suffixes and normalizes.
// "JPMorgan Chase & Co." → "jpmorgan chase"
func normalizeCompany(company string) string {
	company = strings.ToLower(strings.TrimSpace(company))
	for _, suffix := range []string{
		"& co.", "& co", ", inc.", ", inc", " inc.", " inc",
		" corp.", " corp", " corporation", " co.", " co",
		" llc", " ltd", " limited", " group", " holdings",
		" financial", " bancorp", " bancshares",
	} {
		company = strings.TrimSuffix(company, suffix)
	}
	return strings.TrimSpace(company)
}
