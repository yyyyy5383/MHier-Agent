package orchestrator

import "time"

type JobStatus string

const (
	StatusQueued    JobStatus = "queued"
	StatusRunning   JobStatus = "running"
	StatusSucceeded JobStatus = "succeeded"
	StatusFailed    JobStatus = "failed"
	StatusTimeout   JobStatus = "timeout"
	StatusCanceled  JobStatus = "canceled"
)

type StepName string

const (
	StepNone     StepName = "none"
	StepPrepare  StepName = "prepare"
	StepRetrieve StepName = "retrieve"
	StepPredict  StepName = "predict"
	StepFinalize StepName = "finalize"
)

type Config struct {
	Addr               string
	RepoRoot           string
	PythonBin          string
	MaxWorkers         int
	QueueSize          int
	DefaultStepTimeout time.Duration
	KillGrace          time.Duration
}

type JobRequest struct {
	PDFPath        string   `json:"pdf_path"`
	Questions      []string `json:"questions"`
	RunPrepare     *bool    `json:"run_prepare,omitempty"`
	RunRetrieve    *bool    `json:"run_retrieve,omitempty"`
	RunPredict     *bool    `json:"run_predict,omitempty"`
	TimeoutSeconds int      `json:"timeout_seconds,omitempty"`
}

type Job struct {
	ID          string        `json:"job_id"`
	Status      JobStatus     `json:"status"`
	CurrentStep StepName      `json:"current_step"`
	Request     JobRequest    `json:"request"`
	ArtifactDir string        `json:"artifact_dir"`
	CreatedAt   time.Time     `json:"created_at"`
	StartedAt   *time.Time    `json:"started_at,omitempty"`
	FinishedAt  *time.Time    `json:"finished_at,omitempty"`
	Error       string        `json:"error,omitempty"`
	FailedStep  StepName      `json:"failed_step,omitempty"`
	ExitCode    *int          `json:"exit_code,omitempty"`
	Duration    time.Duration `json:"duration,omitempty"`
}

type JobResponse struct {
	JobID     string    `json:"job_id"`
	Status    JobStatus `json:"status"`
	EventsURL string    `json:"events_url"`
	StatusURL string    `json:"status_url"`
}

type Event struct {
	JobID     string    `json:"job_id"`
	Event     string    `json:"event"`
	Step      StepName  `json:"step,omitempty"`
	Status    JobStatus `json:"status,omitempty"`
	Stream    string    `json:"stream,omitempty"`
	Message   string    `json:"message,omitempty"`
	ExitCode  *int      `json:"exit_code,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}
