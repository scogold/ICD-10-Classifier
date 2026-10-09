from icd10.config import BASE, EXPERIMENTS, deep_merge, get_config


def flat_keys(config: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key, value in config.items():
        if isinstance(value, dict):
            keys |= flat_keys(value, f"{prefix}{key}.")
        else:
            keys.add(f"{prefix}{key}")
    return keys


def test_deep_merge_overrides_only_the_given_keys():
    merged = deep_merge({"train": {"lr": 1e-4, "epochs": 3}, "seed": 1}, {"train": {"epochs": 5}})
    assert merged == {"train": {"lr": 1e-4, "epochs": 5}, "seed": 1}


def test_experiments_have_exactly_the_base_keys():
    # Catches typos: an experiment setting a key that BASE doesn't have would silently do nothing.
    for name, config in EXPERIMENTS.items():
        assert flat_keys(config) == flat_keys(BASE), name


def test_get_config_returns_an_independent_copy():
    config = get_config("smoke")
    config["train"]["lr"] = 123.0
    assert EXPERIMENTS["smoke"]["train"]["lr"] != 123.0
