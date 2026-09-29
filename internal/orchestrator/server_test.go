package orchestrator

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestNewServerUsesGinEngine(t *testing.T) {
	handler := NewServer(nil)

	if got, want := reflect.TypeOf(handler).String(), "*gin.Engine"; got != want {
		t.Fatalf("NewServer() handler type = %s, want %s", got, want)
	}
}

func TestHealthRoute(t *testing.T) {
	handler := NewServer(nil)
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	resp := httptest.NewRecorder()

	handler.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("GET /healthz status = %d, want %d", resp.Code, http.StatusOK)
	}
	if got, want := resp.Body.String(), "{\"status\":\"ok\"}\n"; got != want {
		t.Fatalf("GET /healthz body = %q, want %q", got, want)
	}
}
