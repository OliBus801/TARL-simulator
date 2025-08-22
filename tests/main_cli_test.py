from main import main
from src.runner import Runner

def test_dijkstra_eval(monkeypatch):
    calls = []
    def fake_setup(self):
        calls.append("setup")
    def fake_eval(self):
        calls.append("eval")
    monkeypatch.setattr(Runner, "setup", fake_setup)
    monkeypatch.setattr(Runner, "eval", fake_eval)
    main(["--algo", "dijkstra", "--mode", "eval"])
    assert calls == ["setup", "eval"]

def test_mpnn_ppo_train(monkeypatch, tmp_path):
    calls = []
    def fake_setup(self):
        calls.append("setup")
    def fake_train(self):
        calls.append("train")
    def fake_eval(self):
        calls.append("eval")
    monkeypatch.setattr(Runner, "setup", fake_setup)
    monkeypatch.setattr(Runner, "train", fake_train)
    monkeypatch.setattr(Runner, "eval", fake_eval)
    main(["--algo", "mpnn+ppo", "--mode", "train", "--output-dir", str(tmp_path)])
    assert calls == ["setup", "train", "eval"]

def test_mpnn_eval(monkeypatch):
    calls = []
    def fake_setup(self):
        calls.append("setup")
    def fake_eval(self):
        calls.append("eval")
    monkeypatch.setattr(Runner, "setup", fake_setup)
    monkeypatch.setattr(Runner, "eval", fake_eval)
    main(["--algo", "mpnn", "--mode", "eval"])
    assert calls == ["setup", "eval"]


def test_wandb_flag(monkeypatch):
    captured = {}

    def fake_init(self, args):
        captured["wandb"] = args.wandb
        self.args = args

    monkeypatch.setattr(Runner, "__init__", fake_init)
    monkeypatch.setattr(Runner, "setup", lambda self: None)
    monkeypatch.setattr(Runner, "eval", lambda self: None)

    main(["--algo", "dijkstra", "--mode", "eval", "--wandb"])

    assert captured["wandb"] is True
