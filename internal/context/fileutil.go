package context

import (
	"os"
)

func writeFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0644)
}

func readFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(data), nil
}
