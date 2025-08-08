def run_episode(simulator, agents, steps: int = 1):
    """Run a short evaluation episode using classical Dijkstra agents."""
    simulator.agent = agents
    for _ in range(steps):
        simulator.run()
    return simulator
