import textwrap
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

from src.agents.base import Agents


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def tmp_scenario_dir(tmp_path):
    """
    Creates a minimal scenario directory containing 'population.xml' and 'network.xml' files for testing purposes.

    Args:
      tmp_path (pathlib.Path): Temporary directory path provided by pytest's tmp_path fixture.

    Returns:
      pathlib.Path: Path to the created scenario directory containing the XML files.

    Description:
      - The function generates a subdirectory named 'scenario' within the given temporary path.
      - It writes a 'population.xml' file describing four persons, each with a plan consisting of activities and links.
      - It writes a 'network.xml' file describing a simple network with three nodes and three links.
      - Both XML files are written in UTF-8 encoding.
      - This setup is useful for unit tests that require a minimal MATSim-like scenario with population and network data.

    Note:
      The XML content is hardcoded and intended for demonstration or testing only.
    """
    scen = tmp_path / "scenario"
    scen.mkdir(parents=True, exist_ok=True)

    # population.xml
    pop_xml = textwrap.dedent("""\
    <?xml version='1.0' encoding='utf-8'?>
    <population>
      <!-- Person 1 : 3 trips -->
      <person id="1">
        <plan>
          <act type="h" x="-20000" y="0" link="1" end_time="06:00" />
          <act type="w" x="-10000" y="0" link="3" end_time="07:00"/>
          <act type="h" x="-20000" y="0" link="1" end_time="08:00"/>
          <act type="w" x="-20000" y="0" link="3" end_time="09:00"/>
        </plan>
      </person>
      <!-- Person 2 : One trip -->
      <person id="2">
        <plan>
          <act type="h" x="-20000" y="0" link="1" end_time="06:00" />
          <act type="w" x="-10000" y="0" link="3" end_time="07:00"/>
        </plan>
      </person>
      <!-- Person 3 : No trips -->
      <person id="3">
        <plan>
        </plan>
      </person>
      <!-- Person 4 : One Trip at different times and different starting link -->
      <person id="4">
        <plan>
          <act type="h" x="-20000" y="0" link="3" end_time="06:30"/>
          <act type="w" x="-10000" y="0" link="1" end_time="07:00"/>
        </plan>
      </person>
    </population>
    """)
    (scen / "population.xml").write_text(pop_xml, encoding="utf-8")

    # network.xml
    net_xml = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">
    <network name="equil test network">
       <nodes>
          <node id="1" x="-20000" y="0"/>
          <node id="2" x="-15000" y="0"/>
          <node id="3" x="-10000" y="0"/>
       </nodes>
       <links capperiod="01:00:00">
          <link id="1" from="1" to="2" length="25" capacity="1" freespeed="8.33" permlanes="1" />
          <link id="2" from="2" to="3" length="25" capacity="1" freespeed="8.33" permlanes="1" />
          <link id="3" from="3" to="1" length="25" capacity="1" freespeed="8.33" permlanes="1" />
       </links>
    </network>
    """)
    (scen / "network.xml").write_text(net_xml, encoding="utf-8")

    return scen


def _hhmm_to_seconds(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def _expected_trips_from_xml(pop_xml_path: Path):
    """
    Computes the expected trips with SRC/DEST intersection indices.

    - Intersections are enumerated in the sorted order of their identifiers.
    - Each intersection i is converted into two nodes: SRC(i) and DEST(i)
      with indices:
      SRC(i)  = num_links + 2*idx
      DEST(i) = num_links + 2*idx + 1
    - For a trip (act a -> act b):
      origin = 'from' node of the link of act a
      destination = 'to' node of the link of act b
    Returns: [(src_idx:int, dest_idx:int, dep_time:int)]
    """
    net_path = pop_xml_path.with_name("network.xml")
    net_root = ET.parse(net_path).getroot()
    links_el = net_root.find("links")

    link_from = {}
    link_to = {}
    intersections = set()
    for link in links_el:
        lid = link.attrib["id"]
        f = link.attrib["from"]
        t = link.attrib["to"]
        link_from[lid] = f
        link_to[lid] = t
        intersections.update([f, t])

    num_links = len(list(links_el))
    intersection_indices = {
        inter: (num_links + 2 * i, num_links + 2 * i + 1)
        for i, inter in enumerate(sorted(intersections))
    }

    root = ET.parse(pop_xml_path).getroot()
    trips = []
    for person in root.findall("person"):
        acts = person.findall("./plan/act")
        for a, b in zip(acts, acts[1:]):
            origin_node = a.attrib["link"]
            dest_node = b.attrib["link"]
            origin_idx = intersection_indices[origin_node][0]
            dest_idx = intersection_indices[dest_node][1]
            dep_time = _hhmm_to_seconds(a.attrib["end_time"])
            trips.append((origin_idx, dest_idx, dep_time))
    

    return trips


# -----------------------
# Tests
# -----------------------

def test_config_agents_from_xml_basic(tmp_scenario_dir):
    """
    Checks:
    - presence of a dummy (row 0) with a distinctive feature,
    - total number of trips (dummy + XML trips),
    - temporal sorting of departures,
    - (ORIGIN, DESTINATION, DEP_TIME) identical to XML (identity mapping),
    - default attributes (AGE/SEX/EMPLOYMENT_STATUS),
    - no “trip” entry with zero DEP_TIME.
    """
    agent = Agents(device="cpu")

    agent.config_agents_from_xml(str(tmp_scenario_dir))

    feats = agent.agent_features
    h = agent

    assert isinstance(feats, torch.Tensor)
    assert feats.ndim == 2 and feats.size(1) >= len(agent)

    # --- dummy ---
    assert feats[0, h.DEPARTURE_TIME] >= 25 * 3600 or feats[0, h.DONE] == 1 or feats[0, h.ON_WAY] == 0

    # --- Number of trips ---
    expected = _expected_trips_from_xml(tmp_scenario_dir / "population.xml")
    print(expected)
    assert feats.shape[0] == len(expected) + 1, f"Expected {len(expected)} trips + 1 dummy, got {feats.shape[0]} rows"

    real_agents = feats[1:]  # ignore dummy

    # --- exact correspondence ---

    got = [(int(real_agents[i, h.ORIGIN].item()),
            int(real_agents[i, h.DESTINATION].item()),
            int(real_agents[i, h.DEPARTURE_TIME].item()))
           for i in range(real_agents.shape[0])]
    assert got == expected, f"\nExpected (ORIGIN, DEST, DEP): {expected}\nGot: {got}"

    # --- default attributes ---
    assert torch.all(real_agents[:, h.SEX] == 0)
    assert torch.all(real_agents[:, h.EMPLOYMENT_STATUS] == 0)
    assert torch.all(real_agents[:, h.AGE] == 20)

    # --- no zero DEP_TIME ---
    assert (real_agents[:, h.DEPARTURE_TIME] == 0).sum().item() == 0
