import pytest

from src.equinox.sampling.trespass.inference import load_run_configuration, resolve_checkpoint


def test_resolve_checkpoint_picks_latest(tmp_path) -> None:
    case_dir = tmp_path / "case"
    results_dir = case_dir / "batch_sgd_results"
    results_dir.mkdir(parents=True)

    (results_dir / "checkpoint_iter_50.pt").write_text("a", encoding="utf-8")
    (results_dir / "checkpoint_iter_200.pt").write_text("b", encoding="utf-8")

    resolved = resolve_checkpoint(str(case_dir))
    assert resolved.name == "checkpoint_iter_200.pt"


def test_load_run_configuration_filters_unknown_keys(tmp_path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        "cost_model_version: lin_disent\nunknown_key: 1\n",
        encoding="utf-8",
    )

    config = load_run_configuration(str(config_path))
    assert config.cost_model_version == "lin_disent"


def test_load_run_configuration_rejects_legacy(tmp_path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        "cost_model_beta0: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_run_configuration(str(config_path))
