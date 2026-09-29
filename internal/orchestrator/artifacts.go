package orchestrator

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func ensureDir(path string) error {
	return os.MkdirAll(path, 0o755)
}

func writeJSON(path string, value any) error {
	if err := ensureDir(filepath.Dir(path)); err != nil {
		return err
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func appendJSONL(path string, value any) error {
	if err := ensureDir(filepath.Dir(path)); err != nil {
		return err
	}
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(data)
	return err
}

func safeIDPart(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "job"
	}
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "job"
	}
	if len(out) > 48 {
		return out[:48]
	}
	return out
}

func samplesForRequest(req JobRequest) []map[string]string {
	docID := filepath.Base(req.PDFPath)
	if docID == "." || docID == string(filepath.Separator) || docID == "" {
		docID = "unknown.pdf"
	}
	samples := make([]map[string]string, 0, len(req.Questions))
	for _, question := range req.Questions {
		samples = append(samples, map[string]string{
			"doc_id":   docID,
			"question": question,
		})
	}
	return samples
}

func copyIfExists(src, dst string) error {
	info, err := os.Stat(src)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if info.IsDir() {
		return nil
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if err := ensureDir(filepath.Dir(dst)); err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func validateRequest(req JobRequest) error {
	if strings.TrimSpace(req.PDFPath) == "" {
		return fmt.Errorf("pdf_path must not be empty")
	}
	if len(req.Questions) == 0 {
		return fmt.Errorf("questions must not be empty")
	}
	for _, question := range req.Questions {
		if strings.TrimSpace(question) == "" {
			return fmt.Errorf("questions must not contain empty values")
		}
	}
	return nil
}
