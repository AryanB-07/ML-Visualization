"""trajmap: training and unlearning trajectories on one live loss-landscape map."""
from .basis import Basis, BasisManager
from .data import IndexedDataset
from .landscape import Evaluator, GridCache
from .metrics import TrajectoryMetric, compute_metrics, list_metrics, register_metric
from .store import Run, Trajectory
from .tracer import Recorder, Tracer
from .unlearning import UnlearningMethod, list_methods, register_method, run_unlearning

__version__ = "0.1.0"
__all__ = [
    "Tracer", "Recorder", "Run", "Trajectory", "Basis", "BasisManager", "Evaluator", "GridCache",
    "UnlearningMethod", "register_method", "list_methods", "run_unlearning",
    "TrajectoryMetric", "register_metric", "list_metrics", "compute_metrics", "IndexedDataset",
]
