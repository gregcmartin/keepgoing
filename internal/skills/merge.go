package skills

import "encoding/json"

// mergeJSON combines two JSON objects. New keys from b override a,
// but existing keys in a are preserved if b doesn't have them.
func mergeJSON(a, b json.RawMessage) json.RawMessage {
	var mapA, mapB map[string]json.RawMessage

	if err := json.Unmarshal(a, &mapA); err != nil {
		return b // Can't parse a, just use b
	}
	if err := json.Unmarshal(b, &mapB); err != nil {
		return a // Can't parse b, just use a
	}

	// Merge b into a — b's non-empty values win
	for k, v := range mapB {
		// Skip if the new value is null, empty string, or empty array
		sv := string(v)
		if sv == "null" || sv == `""` || sv == "[]" {
			continue
		}
		mapA[k] = v
	}

	result, err := json.Marshal(mapA)
	if err != nil {
		return b
	}
	return result
}
