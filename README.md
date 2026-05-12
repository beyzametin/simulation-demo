# Honest CAN-IDS Simulator — Live Demo

Synthetic CAN-bus traffic generator with eight simulated ECUs, four attacker
profiles (DoS, ID-sweep, replay, spoofing), and a load-dependent
observation-loss model. Each attacker has a five-level intensity ladder
ranging from undetectable design overlap to recorded-bench extremity.

The companion paper (in preparation) re-evaluates CAN intrusion detection
across five public and one benchtop dataset under leakage-controlled splits.
Within-dataset detection saturates everywhere; cross-testbed transfer
collapses mean ROC-AUC from near-saturation down to roughly chance on
several pairs. This simulator surfaces the regime where the
*observation-asymmetry* feature channel — defender-side packet loss and
ID-coverage shrinkage — earns measurable lift over the base feature set, a
result obscured by the saturated bench corpus.

## Live demo

A hosted Streamlit instance is available at *(URL added after first deploy)*.

## Local

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Reproducing the simulator corpus

```bash
python code/15_generate_sim.py            # writes data/processed/sim.parquet
```

This public mirror carries the simulator backend only. The full evaluation
pipeline (parsers for HCRL / ROAD / CAN-MIRGU / can-train-and-test, feature
extraction, cross-testbed training, calibration, figures, tables) is
maintained in a private companion repository pending the paper's
publication.

## License

MIT. See `LICENSE`.
