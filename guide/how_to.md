# Training
- Start with `prep_all.py`, modify for the airport pair.
- Then launch `tres_batch.py` to compute the state transititions.
- Then proceed with `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py` to learn the weights and the preferences.

# Inference
python -c "from equinox.sampling.pipeline import compute_4d_path_for_dataset as f; f(case_dir='data/cases/LGAV_LFPG', flight_id='45CAB5AFR36HN', n_samples=200, policy='sample', return_4d=False, output_dir='data/cases/LGAV_LFPG/inference_flight_45CAB5AFR36HN')"