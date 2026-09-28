# Tests

```bash
python -m unittest discover -s tests -v
```

`test_scoring.py` and `test_api_backends.py` use synthetic records and mocked clients.

`test_runner_golden.py` runs both runners end to end with a fake model under 27 conditions
(`golden/make_goldens.py` lists them) and compares every prompt sent to the model and every
saved record with the snapshots in `golden/snapshots/`. The finance conditions read Conv-FinRe
from the Hugging Face cache with network access switched off, so load the dataset once first:

```bash
python -c "from agent_system.dataset import ConvFinReDataset; ConvFinReDataset().load()"
```

After an intended change to prompts, tools or saved records, rewrite the snapshots with
`python tests/golden/make_goldens.py` and review the diff.
