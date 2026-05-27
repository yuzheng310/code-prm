# Code-PRM

Process Reward Model for Code Agent multi-turn trajectories.

See `docs/superpowers/specs/2026-05-27-code-prm-design.md` for full design.

## Quick Start

```bash
conda env create -f environment.yml
conda activate code-prm
cp .env.example .env  # then fill in keys
pytest tests/         # smoke test
```

## Status

- [ ] Phase 1: Foundation (in progress — code scaffold done, data collection pending)
- [ ] Phase 2: Training
- [ ] Phase 3: Eval & Ship
