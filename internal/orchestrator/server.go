package orchestrator

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

type Server struct {
	manager *Manager
	router  *gin.Engine
}

func NewServer(manager *Manager) http.Handler {
	gin.SetMode(gin.ReleaseMode)

	s := &Server{
		manager: manager,
		router:  gin.New(),
	}
	s.router.Use(gin.Recovery(), s.corsMiddleware())
	s.routes()
	return s.router
}

func (s *Server) routes() {
	s.router.GET("/healthz", s.handleHealth)
	s.router.POST("/api/jobs", s.handleJobs)
	s.router.GET("/api/jobs/:job_id", s.handleGetJob)
	s.router.DELETE("/api/jobs/:job_id", s.handleCancelJob)
	s.router.GET("/api/jobs/:job_id/events", s.handleEvents)
	s.router.OPTIONS("/*path", func(c *gin.Context) {
		c.Status(http.StatusNoContent)
	})
}

func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (s *Server) handleJobs(c *gin.Context) {
	var req JobRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		writeError(c, http.StatusBadRequest, err.Error())
		return
	}
	job, err := s.manager.Submit(req)
	if err != nil {
		writeError(c, http.StatusBadRequest, err.Error())
		return
	}
	c.JSON(http.StatusAccepted, JobResponse{
		JobID:     job.ID,
		Status:    job.Status,
		EventsURL: "/api/jobs/" + job.ID + "/events",
		StatusURL: "/api/jobs/" + job.ID,
	})
}

func (s *Server) handleGetJob(c *gin.Context) {
	jobID := c.Param("job_id")
	job, ok := s.manager.Get(jobID)
	if !ok {
		writeError(c, http.StatusNotFound, "job not found")
		return
	}
	c.JSON(http.StatusOK, job)
}

func (s *Server) handleCancelJob(c *gin.Context) {
	jobID := c.Param("job_id")
	if !s.manager.Cancel(jobID) {
		writeError(c, http.StatusNotFound, "job not found")
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"job_id": jobID, "status": string(StatusCanceled)})
}

func (s *Server) handleEvents(c *gin.Context) {
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		writeError(c, http.StatusInternalServerError, "streaming unsupported")
		return
	}

	jobID := c.Param("job_id")
	events, cleanup, ok := s.manager.Subscribe(jobID)
	if !ok {
		writeError(c, http.StatusNotFound, "job not found")
		return
	}
	defer cleanup()

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	fmt.Fprintf(c.Writer, "event: connected\n")
	fmt.Fprintf(c.Writer, "data: {\"job_id\":%q}\n\n", jobID)
	flusher.Flush()

	for {
		select {
		case <-c.Request.Context().Done():
			return
		case event, ok := <-events:
			if !ok {
				return
			}
			data, _ := json.Marshal(event)
			fmt.Fprintf(c.Writer, "event: %s\n", event.Event)
			fmt.Fprintf(c.Writer, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func (s *Server) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Accept")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

func writeError(c *gin.Context, status int, message string) {
	c.JSON(status, gin.H{"error": message})
}
