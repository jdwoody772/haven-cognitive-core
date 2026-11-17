# Haven Cognitive Core — experimental v0.1

An initial **non-LLM** persistent adaptive graph, with a mobile-friendly web interface.

## Run

Requires Python 3.10+; no third-party dependencies.

```bash
python server.py
```

Open http://127.0.0.1:8765 on the host computer. To reach it from an iPhone on the same trusted LAN, run with `HAVEN_HOST=0.0.0.0` and browse to the computer's LAN address on port 8765. **The prototype has no authentication. Do not expose it directly to the public Internet.** Use a secured, authenticated tunnel or reverse proxy for remote access.

Run tests with `python -m unittest -v`.

## What exists

- Persistent weighted directed graph of labeled nodes
- Numerical activation and recurrent propagation
- Online prediction-error weight updates
- Creation of nodes when new symbols are taught
- Interactive browser interface showing real graph state
- Tests for learning, reversal and persistence

## What does not yet exist

This is **not sentient**, not a general intelligence, and not a biological brain replica. The labels are supplied by the teacher. Node vectors are randomly initialized and are **not learned embeddings**. There is no genuine autonomous goal selection, concept discovery, structural pruning, episodic consolidation, robust continual learning, or validated generalization. The recurring loop propagates activation but does not independently select actions. Prediction ties initially depend on node insertion order. These limitations are intentional and must not be mistaken for achievements.

## Next research milestone

Introduce a trainable distributed representation and a recurrent predictive model that generalizes to held-out compositions, benchmarked against lookup-table memorization. Then add a controlled action environment, replay/consolidation, and measurable autonomous exploration. Preserve ablation tests and transparent reporting.
