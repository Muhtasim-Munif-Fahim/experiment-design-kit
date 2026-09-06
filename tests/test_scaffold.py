"""Scaffold tests for the package surface."""


from experiment_design_kit import __version__


def test_version_is_semver() -> None:
    assert __version__ == "0.1.0"
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_version_exposed() -> None:
    import experiment_design_kit as edk

    assert edk.__version__
