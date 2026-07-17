"""Tests for config.exceptions."""

from __future__ import annotations

import pytest

from config.exceptions import (
    ConfigurationError,
    ConfigurationFileError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [
        MissingEnvironmentVariableError,
        InvalidConfigurationError,
        ConfigurationFileError,
    ],
)
def test_subclasses_inherit_from_configuration_error(exc_cls: type[Exception]) -> None:
    """Each specific exception must be a ConfigurationError."""
    assert issubclass(exc_cls, ConfigurationError)


def test_configuration_error_is_an_exception() -> None:
    """The base class must itself be a normal Exception."""
    assert issubclass(ConfigurationError, Exception)


@pytest.mark.parametrize(
    "exc_cls",
    [
        ConfigurationError,
        MissingEnvironmentVariableError,
        InvalidConfigurationError,
        ConfigurationFileError,
    ],
)
def test_can_be_raised_and_caught_with_message(exc_cls: type[Exception]) -> None:
    """Each exception can be raised and caught, carrying a message."""
    with pytest.raises(exc_cls) as exc_info:
        raise exc_cls("something went wrong")
    assert "something went wrong" in str(exc_info.value)


@pytest.mark.parametrize(
    "exc_cls",
    [
        MissingEnvironmentVariableError,
        InvalidConfigurationError,
        ConfigurationFileError,
    ],
)
def test_subclasses_are_catchable_as_base_class(exc_cls: type[Exception]) -> None:
    """Catching ConfigurationError must catch any specific subtype."""
    with pytest.raises(ConfigurationError):
        raise exc_cls("failure")


def test_subclasses_are_distinct_from_each_other() -> None:
    """Different subtypes must not be interchangeable with one another."""
    assert not issubclass(MissingEnvironmentVariableError, InvalidConfigurationError)
    assert not issubclass(InvalidConfigurationError, MissingEnvironmentVariableError)
    assert not issubclass(ConfigurationFileError, InvalidConfigurationError)
