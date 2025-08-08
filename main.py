# The file where the main experiments will be done

from src import TransportationSimulator, Agents, DijkstraAgents
from src.feature_helpers import FeatureHelpers
import time
from tqdm import tqdm
import torch
import argparse
import os
import sys

def end_sequence():
    end = time.time()
    mask = agents.agent_features[:, agents.DONE] == 1
    average_travel = torch.mean(agents.agent_features[mask, agents.ARRIVAL_TIME] - agents.agent_features[mask, agents.DEPARTURE_TIME] )

    print(f"Temps trajet moyen : {average_travel}")
    print(f"Temps d'exécution : {end-begin}")
    print(f"Temps d'insertion des agents : {simulator.inserting_time}")

    print(f"Temps de calculs des directions des agents : {simulator.choice_time}")

    print(f"Temps modèle des agents : {simulator.core_time}")

    print(f"Temps de retraits des agents : {simulator.withdraw_time}")

    
    if args.save_outputs:
        simulator.plot_computation_time(args.output_dir)
        simulator.compute_node_metrics(args.output_dir)

    simulator.plot_leg_histogram(args.save_outputs, args.output_dir)
    simulator.plot_road_optimality([i for i in range(10, 19)], args.save_outputs, args.output_dir)


# Add argument parsing for command line options (timestep)
parser = argparse.ArgumentParser(description="Run the transportation simulation.")
parser.add_argument("--timestep", type=int, default=6, help="Time step for the simulation in seconds.")
parser.add_argument("--save_outputs", action="store_true", help="Save the outputs of the simulation. Default is False.")
parser.add_argument("--network_file", type=str, default="Easy_network_PT", help="Path to the network XML file. Should be stored in the data folder.")
parser.add_argument("--population_file", type=str, default="Easy_population_PT", help="Path to the population XML file. Should be stored in the data folder.")
parser.add_argument("--output_dir", type=str, default="data/outputs", help="Directory to save the output files. Default is 'data/outputs'.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")  # Force to use CPU for debugging
print("Using device:", device)

if torch.cuda.is_available():
    print("Nombre de GPU disponibles :", torch.cuda.device_count())
    print("GPU courant :", torch.cuda.current_device())

#agents = Agents()
agents = DijkstraAgents(device)
simulator = TransportationSimulator(device)

# Load the network from XML file or from a saved file
try:
    simulator.load_network(os.path.join("save", args.network_file + ".pt"))
    print("Network loaded from save file.")
except FileNotFoundError:
    print("Save not found, creating network from XML.")
    simulator.config_network(os.path.join("data", args.network_file + ".xml.gz"))
    simulator.save_network(os.path.join("save", args.network_file + ".pt"))

# Load agents from XML file or from a saved file
try: 
    agents.load(os.path.join("save", args.population_file + ".pt"))
    print("Agents loaded from save file.")
except FileNotFoundError:
    print("Save not found, creating agents from XML.")
    agents.config_agents_from_xml(os.path.join("data", args.population_file + ".xml.gz"), os.path.join("data", args.network_file + ".xml.gz"))
    agents.save(os.path.join("save", args.population_file + ".pt")) 

simulator.agent = agents

# Define the simulation start time (in seconds) (6 AM)
simulator.config_parameters(timestep=args.timestep, start_time= 6 * 3600, leg_histogram=True, road_optimality=True, node_metrics=True)
simulator.configure_core()

h = FeatureHelpers(simulator.Nmax)
print("Simulator maximum number of agents:", simulator.Nmax)

begin = time.time()

# Calculate the number of timestep to run based on the total simulation time
# 86400 seconds in a day minus the start time divided by the timestep
n_timestep = (86400 - simulator.time) // args.timestep
try:
    for i in tqdm(range(n_timestep), desc="Simulation Progress", unit="timestep"):
        # Run the scenario
        #time.sleep(0.05)
        #print(simulator.graph.x[:, h.NUMBER_OF_AGENT]) # Uncomment for debugging

        simulator.run()
except KeyboardInterrupt:
    print("Simulation interrupted by user.")
    end_sequence()
    # System exit to avoid further error catching
    sys.exit(0)

except Exception as e:
    print(e)
    end_sequence()
    # System exit to avoid further errors
    sys.exit(1)

end_sequence()
