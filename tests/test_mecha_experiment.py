"""Developer preflight of the canonical synthetic MECHA experiment.

Runs the same driver and the same public CLI commands as
``python -m atlas_interop demo --workflows mecha``, at a tiny size with the
``preflight`` preset. This is a technical check only; it is not the
reviewer-facing experiment and its numbers mean nothing.
"""

from atlas_interop import demo
from atlas_interop import mecha_experiment as X


def test_mecha_experiment_preflight(tmp):
    log = []
    res = X.run_experiment(f"{tmp}/pre", cli=lambda a: demo._cli(a, log), check=demo.check,
                           counts=dict(reference=4, meta_training=200, validation=40, support=60, test=50),
                           preset="preflight", draw_seeds=[11, 12], rank=8)
    assert res["technically_valid"]
    cmds = [c["command"].split()[3] for c in log]
    assert cmds == ["fit-mapping", "mecha-make-splits", "mecha-train", "mecha-adapt", "mecha-compare"]
    assert all(c["returncode"] == 0 for c in log)
