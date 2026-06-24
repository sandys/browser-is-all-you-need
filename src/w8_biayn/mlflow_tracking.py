"""Optional MLflow tracking helpers for non-failing benchmark integrations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


def parse_mlflow_tags(values: list[str] | tuple[str, ...] | None) -> tuple[tuple[str, str], ...]:
    """Parse repeated ``key=value`` tags into a stable tuple.

    Raises
    ------
    ValueError
        If the tag string does not contain '=' or has an empty key.
    """
    if not values:
        return ()
    parsed: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"MLflow tag must be key=value: {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"MLflow tag must have a non-empty key: {raw!r}")
        parsed.append((key, value))
    return tuple(parsed)


@dataclass(frozen=True)
class MlflowRunInfo:
    """Normalized MLflow tracking identifiers for a single benchmark run."""

    enabled: bool
    run_id: str | None = None
    run_name: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    tracking_uri: str | None = None


class MlflowTracker:
    """Thin context wrapper for MLflow start/stop with no-op compatibility."""

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str | None,
        run_name: str | None,
        tags: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = tags
        self._run: Any | None = None
        self._info: MlflowRunInfo = MlflowRunInfo(enabled=False)

    @property
    def run_info(self) -> MlflowRunInfo:
        return self._info

    @property
    def is_active(self) -> bool:
        return self._info.enabled and self._run is not None

    def __enter__(self) -> "MlflowTracker":
        if self._tracking_uri is None and self._experiment_name is None and self._run_name is None:
            return self
        try:
            import mlflow
        except ImportError:
            print("MLflow is not installed; skipping MLflow logging.")
            self._info = MlflowRunInfo(enabled=False)
            return self

        try:
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            experiment_id = None
            if self._experiment_name:
                mlflow.set_experiment(self._experiment_name)
                experiment = mlflow.get_experiment_by_name(self._experiment_name)
                experiment_id = experiment.experiment_id if experiment is not None else None
            self._run = mlflow.start_run(run_name=self._run_name)
            for key, value in self._tags:
                mlflow.set_tag(key, value)
            run_id = self._run.info.run_id if self._run is not None and self._run.info else None
            self._info = MlflowRunInfo(
                enabled=True,
                run_id=run_id,
                run_name=self._run_name,
                experiment_id=experiment_id,
                experiment_name=self._experiment_name,
                tracking_uri=self._tracking_uri,
            )
            return self
        except Exception as exc:  # pragma: no cover - depends on environment
            print(
                f"Warning: could not initialize MLflow ({type(exc).__name__}); "
                "skipping MLflow logging."
            )
            self._info = MlflowRunInfo(enabled=False)
            return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            import mlflow
        except Exception:
            return None
        if self._run is not None:
            mlflow.end_run()
            self._run = None
        return None

    def log_param(self, key: str, value: Any) -> None:
        if not self.is_active:
            return None
        try:
            import mlflow
            mlflow.log_param(key, value)
        except Exception:
            return None

    def log_metric(self, key: str, value: float) -> None:
        if not self.is_active:
            return None
        try:
            import mlflow
            mlflow.log_metric(key, value)
        except Exception:
            return None


@contextmanager
def ensure_tracker(
    *,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_tags: tuple[tuple[str, str], ...] | None = None,
) -> Iterator[MlflowTracker]:
    """Create a tracker context for one benchmark invocation."""
    tracker = MlflowTracker(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        run_name=mlflow_run_name,
        tags=mlflow_tags or (),
    )
    with_tracker = tracker.__enter__()
    try:
        yield with_tracker
    finally:
        tracker.__exit__(None, None, None)
