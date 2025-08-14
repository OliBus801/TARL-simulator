import os
import torch
import pytest

from src.agents.base import Agents


def build_files(base_path):
    data_dir = base_path / "data" / "scenario"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "network.xml").write_text(
        """
<network>
  <nodes>
    <node id='A1' x='0' y='0'/>
    <node id='A2' x='0' y='0'/>
    <node id='B1' x='100' y='0'/>
    <node id='B2' x='100' y='0'/>
    <node id='C1' x='200' y='0'/>
    <node id='C2' x='200' y='0'/>
    <node id='D1' x='0' y='100'/>
    <node id='D2' x='0' y='100'/>
    <node id='E1' x='100' y='100'/>
    <node id='E2' x='100' y='100'/>
  </nodes>
  <links>
    <link id='0' from='A1' to='A2'/>
    <link id='1' from='B1' to='B2'/>
    <link id='2' from='C1' to='C2'/>
    <link id='3' from='D1' to='D2'/>
    <link id='4' from='E1' to='E2'/>
  </links>
</network>
"""
    )
    (data_dir / "population.xml").write_text(
        """
<population>
  <person id='A'>
    <attributes>
      <attribute name='carAvail'>always</attribute>
      <attribute name='sex'>f</attribute>
      <attribute name='employed'>yes</attribute>
      <attribute name='age'>40</attribute>
    </attributes>
    <plan>
      <act type='home' x='0' y='0' end_time='08:30:00'/>
      <act type='work' x='100' y='0' end_time='17:00:00'/>
      <act type='home' x='200' y='0'/>
    </plan>
  </person>
  <person id='B'>
    <attributes>
      <attribute name='car_avail'>always</attribute>
    </attributes>
    <plan>
      <act type='home' x='0' y='100' end_time='09:15'/>
      <act type='shop' x='100' y='100'/>
    </plan>
  </person>
  <person id='C'>
    <attributes>
      <attribute name='carAvail'>never</attribute>
    </attributes>
    <plan>
      <act type='home' x='0' y='0' end_time='07:00:00'/>
      <act type='work' x='100' y='0'/>
    </plan>
  </person>
</population>
"""
    )
    return data_dir


@pytest.fixture
def scenario_path(tmp_path):
    return build_files(tmp_path)


def test_config_agents_from_xml(scenario_path, monkeypatch):
    agent = Agents(device='cpu')
    monkeypatch.chdir(scenario_path.parent.parent)  # change to tmp base directory
    agent.config_agents_from_xml('scenario')
    assert agent.agent_features.dtype == torch.float32
    assert agent.agent_features.device.type == 'cpu'
    # two agents selected -> 3 trips (2+1)
    assert agent.agent_features.shape[0] == 3
    assert agent.agent_features[0, agent.DEPARTURE_TIME] == 25*3600
    # second trip departure time
    assert agent.agent_features[1, agent.DEPARTURE_TIME] == 17*3600
    assert agent.agent_features[2, agent.DEPARTURE_TIME] == 9*3600 + 15*60
    # default attributes for agent B
    assert agent.agent_features[2, agent.SEX] == 0
    assert agent.agent_features[2, agent.EMPLOYMENT_STATUS] == 0
    assert agent.agent_features[2, agent.AGE] == 20


def test_verbose_output(scenario_path, monkeypatch, capsys):
    agent = Agents(device='cpu')
    monkeypatch.chdir(scenario_path.parent.parent)
    agent.config_agents_from_xml('scenario', verbose=True)
    captured = capsys.readouterr().out
    assert 'Trips per agent' in captured
    assert 'Exclusion reasons' in captured
