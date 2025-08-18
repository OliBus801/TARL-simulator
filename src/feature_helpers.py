class FeatureHelpers:
    """
    A class to help with feature indexing and manipulation for MPNN models.

    Features:
        AGENT_POSITION: slice(0, Nmax)
            - The FIFO queue. Contains the IDs of agents.
        AGENT_TIME_ARRIVAL: slice(Nmax, 2 * Nmax)
            - The arrival times of agents in the FIFO queue.
        AGENT_POSITION_AT_ARRIVAL: slice(2 * Nmax, 3 * Nmax)
            - The positions of agents at their arrival.
        MAX_NUMBER_OF_AGENT: 3 * Nmax
            - The size of the FIFO queue.
        NUMBER_OF_AGENT: 3 * Nmax + 1
            - The current number of agents on the road.
        FREE_FLOW_TIME_TRAVEL: 3 * Nmax + 2
            - The free flow travel time on that road.
        LENGHT_OF_ROAD: 3 * Nmax + 3
            - The length of the road.
        MAX_FLOW: 3 * Nmax + 4
            - The maximum flow capacity of the road.
        SELECTED_ROAD: 3 * Nmax + 5
            - The index of the selected road. (?)
        ROAD_INDEX: 3 * Nmax + 6
            - The index of the road.
        HEAD_FIFO: 0
            - The ID of the agent at the head of the FIFO queue.
        HEAD_FIFO_TIME: Nmax
            - The time at the head of the FIFO queue. (?)
        HEAD_FIFO_CONG: 2*Nmax
            - The congestion value at the head of the FIFO queue. (?)
        CONGESTION_FILE: 3
            - A constant that determines the congestion buffer size reserved only to resolve gridlock situations.
    """

    def __init__(self, Nmax=100):
        self.Nmax = Nmax
        self.AGENT_POSITION = slice(0, Nmax)
        self.AGENT_TIME_ARRIVAL = slice(Nmax, 2 * Nmax)
        self.AGENT_POSITION_AT_ARRIVAL = slice(2 * Nmax, 3 * Nmax)
        self.MAX_NUMBER_OF_AGENT = 3 * Nmax
        self.NUMBER_OF_AGENT = 3 * Nmax + 1
        self.FREE_FLOW_TIME_TRAVEL = 3 * Nmax + 2
        self.LENGHT_OF_ROAD = 3 * Nmax + 3
        self.MAX_FLOW = 3 * Nmax + 4
        self.SELECTED_ROAD = 3 * Nmax + 5
        self.ROAD_INDEX = 3 * Nmax + 6
        self.HEAD_FIFO = 0
        self.HEAD_FIFO_TIME = Nmax
        self.HEAD_FIFO_CONG = 2*Nmax
        self.CONGESTION_FILE = 3 

class AgentFeatureHelpers:
    """ A class to help with agent feature indexing """

    def __init__(self):
        self.ORIGIN = 0                     # Contains the index of the origin road 
        self.DESTINATION = 1                # Contains the index of the destination road
        self.DEPARTURE_TIME = 2             # Contains the index of departure time
        self.ARRIVAL_TIME = 3               # Contains the arrival time in the scenario
        self.AGE = 4                        # Contains the age of the agent
        self.SEX = 5                        # Contains the sex of the agent
        self.EMPLOYMENT_STATUS = 6          # Contains 1 if the agent is employed
        self.ON_WAY = 7                     # Contains the agent is on the road  
        self.DONE = 8                       # Contains the agent  

    def __len__(self):
        return 9
    

class ObservationFeatureHelpers:
    """ A class to help with observation feature indexing """

    def __init__(self):
        self.MAX_NUMBER_OF_AGENT = 0
        self.NUMBER_OF_AGENT = 1
        self.FREE_FLOW_TIME_TRAVEL = 2
        self.LENGHT_OF_ROAD = 3
        self.MAX_FLOW = 4
        self.SELECTED_ROAD = 5
        self.ROAD_INDEX = 6
        self.ORIGIN = 7                     # Contains the index of the origin road 
        self.DESTINATION = 8                # Contains the index of the destination road
        self.DEPARTURE_TIME = 9             # Contains the index of departure time
        self.ARRIVAL_TIME = 10               # Contains the arrival time in the scenario
        self.AGE = 11                        # Contains the age of the agent
        self.SEX = 12                        # Contains the sex of the agent
        self.EMPLOYMENT_STATUS = 13          # Contains 1 if the agent is employed
        self.ON_WAY = 14                     # Contains the agent is on the road  
        self.DONE = 15  