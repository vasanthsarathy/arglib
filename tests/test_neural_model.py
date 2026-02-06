import pytest

from arglib.ai.neural_model import require_transformers, transformers_available


def test_require_transformers_behavior():
    if transformers_available():
        require_transformers()
    else:
        with pytest.raises(RuntimeError):
            require_transformers()
