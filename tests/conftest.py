import pytest


@pytest.fixture
def dir(tmpdir_factory):
    return tmpdir_factory.mktemp("temp")
