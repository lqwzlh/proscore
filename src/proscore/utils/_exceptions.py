class ProScoreError(Exception):
    """Base exception for all proscore errors."""


class DataError(ProScoreError):
    """Raised when input data fails validation."""


class BinningError(ProScoreError):
    """Raised when binning fails."""


class SelectionError(ProScoreError):
    """Raised during feature selection."""
