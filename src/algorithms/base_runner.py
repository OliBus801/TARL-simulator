from tqdm import tqdm

def run_episode(simulator, agents, steps: int = 86400):
    """Run a short evaluation episode using classical agents."""
    simulator.agent = agents
    for _ in tqdm(range(steps), desc="Running Simulation", unit="step"):
        simulator.run()
    return simulator