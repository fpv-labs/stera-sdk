"""Evaluate: compute metrics for a session and render an interactive HTML report."""

from __future__ import annotations

import logging
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Evaluate:
    """Compute quality / quantity metrics for a recorded session and emit a report.

    Usage::

        from stera import Evaluate
        Evaluate(session).show()
        Evaluate(session).export("report.html")

    All inputs are pulled implicitly from the session (camera poses, IMU, mesh,
    point cloud, depth, plus any hand poses added via ``session.add_hand_pose``).

    Pass an ``EvaluateConfig`` to tune health-score thresholds / weights::

        from stera import Evaluate, EvaluateConfig
        cfg = EvaluateConfig(hand_2_target=40, sync_target=95)
        Evaluate(session, config=cfg).show()
    """

    def __init__(self, session, skeleton=None, config=None):
        from stera.eval.config import EvaluateConfig
        self._session = session
        self._skeleton = skeleton
        self._config = config or EvaluateConfig()
        self._metrics: Optional[dict] = None

    def compute(self) -> dict:
        """Crunch all metrics. Cached on the instance."""
        from stera.eval.metrics import compute_all
        if self._metrics is None:
            logger.info("Computing evaluation metrics")
            self._metrics = compute_all(
                self._session, skeleton=self._skeleton, config=self._config,
            )
        return self._metrics

    def export(self, path: str | Path) -> str:
        """Write a self-contained interactive HTML report to ``path``."""
        from stera.eval.report import write_report
        metrics = self.compute()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_report(out, metrics, session=self._session)
        logger.info("Evaluation report saved to %s", out)
        return str(out)

    def show(self) -> str:
        """Write the report to a temp file and open it in the default browser."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="stera_eval_",
        )
        tmp.close()
        path = self.export(tmp.name)
        webbrowser.open(f"file://{Path(path).resolve()}")
        return path
