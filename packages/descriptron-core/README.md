# descriptron-core

Evidence-tiered species descriptions, keys and audits from phenomic matrices.

**Pure pip: no torch, no CUDA, no compiler.** Installs in about a minute on
Linux, macOS and Windows. With the deposited COCO files it reproduces every
analysis in the methods paper without a model of any kind — the audit, key,
matrix, novelty scoring and delimitation are all deterministic.

```bash
pip install descriptron-core
descriptron --list                 # the 62 programs in this package
descriptron biorag_key_builder_v1 --help
biorag-audit --help
```

Writing treatments is the only step that calls a language model:

```bash
pip install "descriptron-core[llm]"
```

For mask prediction, landmark transfer and scale-bar reading, add
[`descriptron-vision`](https://pypi.org/project/descriptron-vision/); for the
annotation GUI, [`descriptron-gui`](https://pypi.org/project/descriptron-gui/).

The programs are shipped as data and executed with `runpy`, exactly as they run
from a shell, so packaging cannot quietly change what they do.
