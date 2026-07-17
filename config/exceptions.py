"""Exception hierarchy for configuration-related errors.

Defines the error types raised when reading, validating, or parsing
project configuration.
"""

from __future__ import annotations


class ConfigurationError(Exception):
    """Base class for all configuration-related errors.

    Catch this to handle any configuration failure generically, without
    caring which specific subtype occurred.
    """


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when a required environment variable is not set.

    Use this when a setting has no default and configuration cannot
    proceed without it (e.g. a required API key or connection string).
    """


class InvalidConfigurationError(ConfigurationError):
    """Raised when a configuration value is present but invalid.

    Use this when a value fails validation or parsing — for example, an
    out-of-range number or an unrecognized enum-like string.
    """


class ConfigurationFileError(ConfigurationError):
    """Raised when a configuration file cannot be read or parsed.

    Use this for filesystem or syntax problems (missing file, malformed
    YAML/TOML/JSON, permission errors) encountered while loading config
    from disk.
    """
