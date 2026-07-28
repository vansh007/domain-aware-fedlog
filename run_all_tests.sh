#!/usr/bin/env bash
# Sanity-check the whole scaffold. All of these run WITHOUT real data.
set -e
echo "== vocabulary ==";     python -m src.vocabulary
echo; echo "== metrics ==";  python -m src.metrics
echo; echo "== length (H1 mechanism) =="; python -m src.methods.length
echo; echo "== known_events (H3 growth) =="; python -m src.methods.known_events
echo; echo "== dataloader demo =="; python -m src.dataloader --demo
echo; echo "== end-to-end smoke test =="; python scripts/smoke_test.py
echo; echo "ALL SCAFFOLD TESTS PASSED (synthetic only — no real results produced)"
