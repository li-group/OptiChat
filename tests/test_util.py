import os, sys
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
from src.Util import *

def test_openai_api_key():
    assert "OPENAI_API_KEY" in os.environ, "You need to set the OPENAI_API_KEY environment variable"

def test_openai_api_key_valid():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    try:
        response = get_completion_standalone("hello", "gpt-3.5-turbo")
        assert isinstance(response, str)
    except Exception as e:
        pytest.fail(f"API error OPENAI_API_KEY is invalid: {e}")

def test_openai_api_key_valid_gpt4():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    try:
        response = get_completion_standalone("hello", "gpt-4")
        assert isinstance(response, str)
    except Exception as e:
        pytest.fail(f"API error OPENAI_API_KEY is invalid for gpt-4: {e}")





