import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("colab_worker", REPO / "scripts/colab_worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def _args(tmp_path, gpu=False):
    return SimpleNamespace(project=str(tmp_path), gpu=gpu)


def _prepare(tmp_path, root, rec):
    d = tmp_path / root / rec
    d.mkdir(parents=True)
    (d / "windows.json").write_text("[]")


def _done(tmp_path, job_id):
    d = tmp_path / "runs/worker" / job_id
    d.mkdir(parents=True)
    (d / "DONE").write_text("{}")


def test_requires_prepared_checks_the_jobs_runs_root(tmp_path):
    job = {"id": "x", "type": "experiment", "requires_prepared": ["s0"], "config": {"runs_root": "runs/brno_bottom"}}
    _prepare(tmp_path, "runs/brno", "s0")  # prepared in the default root only
    assert "brno_bottom" in worker.blocked(job, _args(tmp_path))
    _prepare(tmp_path, "runs/brno_bottom", "s0")
    assert worker.blocked(job, _args(tmp_path)) is None


def test_requires_done_and_ensemble_members(tmp_path):
    final = {"id": "final", "type": "experiment", "requires_done": ["cv0", "cv1"], "config": {}}
    ens = {"id": "ens", "type": "experiment", "config": {"ensemble_of": ["final"]}}
    _done(tmp_path, "cv0")
    assert "cv1" in worker.blocked(final, _args(tmp_path))
    _done(tmp_path, "cv1")
    assert worker.blocked(final, _args(tmp_path)) is None
    assert "final" in worker.blocked(ens, _args(tmp_path))


def test_hold_and_gpu(tmp_path):
    assert worker.blocked({"id": "h", "type": "shell", "hold": True}, _args(tmp_path, gpu=True)) == "on hold"
    assert worker.blocked({"id": "p", "type": "prepare"}, _args(tmp_path)) == "no GPU"
    assert worker.blocked({"id": "p", "type": "prepare"}, _args(tmp_path, gpu=True)) is None


def test_queue_holds_every_final_test_run():
    queue = json.loads((REPO / "experiments/queue.json").read_text())
    ids = {j["id"] for j in queue}
    finals = [j for j in queue if "final" in j["id"] and j["id"] >= "exp_090"]
    assert finals
    for j in finals:  # test-set scoring waits for its CV and a human decision
        assert j.get("hold"), j["id"]
        assert set(j["requires_done"]) <= ids
