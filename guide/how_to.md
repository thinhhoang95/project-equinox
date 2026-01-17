# Training
- Start with `prep_all.py`, modify for the airport pair.
- Then launch `tres_batch.py` to compute the state transititions.
- Then proceed with `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py` to learn the weights and the preferences.