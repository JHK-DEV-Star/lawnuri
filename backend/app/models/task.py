"""
Async task management for LawNuri.

Provides a thread-safe TaskManager singleton for tracking long-running
background tasks (e.g., debate analysis, document processing) with
progress reporting and lifecycle management.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class TaskStatus(str, Enum):
    """Possible states of an async task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a trackable async task with progress information."""

    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialize task state to a dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskManager:
    """
    Thread-safe singleton for managing async tasks.

    Usage:
        manager = TaskManager()
        task = manager.create_task("analysis")
        manager.update_task(task.task_id, progress=0.5, message="Halfway done")
        manager.complete_task(task.task_id, result={"key": "value"})
    """

    _instance: Optional["TaskManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "TaskManager":
        """Ensure only one TaskManager instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._tasks: dict[str, Task] = {}
                    instance._tasks_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def create_task(self, task_type: str = "", message: str = "") -> Task:
        """Create a new task and register it in the manager."""
        task = Task(task_type=task_type, message=message)
        with self._tasks_lock:
            self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by its ID. Returns None if not found."""
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> Optional[Task]:
        """Update task progress, message, or status."""
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if progress is not None:
                task.progress = min(max(progress, 0.0), 1.0)
            if message is not None:
                task.message = message
            if status is not None:
                task.status = status
            task.updated_at = time.time()
            return task

    def complete_task(self, task_id: str, result: Any = None) -> Optional[Task]:
        """Mark a task as completed with an optional result."""
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.result = result
            task.message = "Completed"
            task.updated_at = time.time()
            return task

    def fail_task(self, task_id: str, error: str = "") -> Optional[Task]:
        """Mark a task as failed with an error message."""
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.FAILED
            task.error = error
            task.message = f"Failed: {error}"
            task.updated_at = time.time()
            return task

    def list_tasks(self, task_type: Optional[str] = None) -> list[dict]:
        """
        List all tasks, optionally filtered by type.
        Returns a list of task dictionaries.
        """
        with self._tasks_lock:
            tasks = self._tasks.values()
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [t.to_dict() for t in tasks]

    def cleanup(self, max_age_seconds: float = 3600) -> int:
        """
        Remove completed or failed tasks older than max_age_seconds.
        Returns the number of tasks removed.
        """
        cutoff = time.time() - max_age_seconds
        removed = 0
        with self._tasks_lock:
            to_remove = [
                tid
                for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                and task.updated_at < cutoff
            ]
            for tid in to_remove:
                del self._tasks[tid]
                removed += 1
        return removed


# Module-level convenience instance
task_manager = TaskManager()
