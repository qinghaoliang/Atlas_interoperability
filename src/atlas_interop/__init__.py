"""Atlas-interoperability layer for connectome learning.

Public toolkit accompanying the manuscript "A lightweight atlas-interoperability
layer for connectome learning across heterogeneous brain atlases".

Subpackages
-----------
transform   Atlas-to-atlas mapping fit (entropic optimal transport on paired
            regional time series) and signed low-rank spectral transformation
            of FC/SC connectomes.
flecha      Federated learning (FedAvg) across sites in heterogeneous atlases,
            plus the manuscript's simulated-site experiment.
mecha       Meta-matching across atlas spaces: multi-output DNN trained on
            transformed connectomes, ridge adaptation on support sets, and
            support-only baselines.
evaluation  Reconstruction similarity and native vs cross-atlas ridge
            prediction pipelines, plus analysis summaries.

Command-line entry point: ``python -m atlas_interop <command> --help``.
"""

__version__ = "0.1.0"
