"""Helper utilities for running classical agent simulations."""

from pathlib import Path
import cProfile
import io
import pstats
from typing import Optional, Union

from tqdm import tqdm


def run_episode(
    simulator,
    agents,
    steps: int = 86400,
    profile: bool = False,
    profile_output: Optional[Union[str, Path]] = None,
):
    """Run a short evaluation episode using classical agents.

    Parameters
    ----------
    simulator:
        The simulation engine to step through.
    agents:
        Agent controller making route choices.
    steps: int, optional
        Number of simulation steps to execute, by default ``86400``.
    profile: bool, optional
        If ``True``, wraps the episode in ``cProfile`` and prints statistics.
    profile_output: str or Path, optional
        When given, write the profiling report to this path.
    """

    simulator.agent = agents
    print("\n" + "=" * 10 + " 🚀 Starting Simulation" + "=" * 10)

    profiler = cProfile.Profile() if profile else None
    if profiler is not None:
        profiler.enable()

    for _ in tqdm(range(steps), desc="Running Simulation", unit="step"):
        simulator.step()

    if profiler is not None:
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(20)
        print("\n=== Profiling Results ===")
        print(stream.getvalue())
        if profile_output:
            output_path = Path(profile_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(stream.getvalue())

    return simulator
