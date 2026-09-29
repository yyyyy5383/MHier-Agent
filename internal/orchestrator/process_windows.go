//go:build windows

package orchestrator

import (
	"fmt"
	"os/exec"
	"syscall"
	"time"
)

func prepareProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

func killProcessTree(pid int, grace time.Duration) {
	if pid <= 0 {
		return
	}
	_ = exec.Command("taskkill", "/T", "/F", "/PID", fmt.Sprintf("%d", pid)).Run()
}
