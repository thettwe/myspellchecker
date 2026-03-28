"""Command handlers for the myspellchecker CLI."""

from myspellchecker.commands.build import _cmd_build, validate_build_inputs
from myspellchecker.commands.check import _cmd_check
from myspellchecker.commands.completion import _cmd_completion
from myspellchecker.commands.config_cmd import _cmd_config
from myspellchecker.commands.infer_pos import _cmd_infer_pos
from myspellchecker.commands.segment import _cmd_segment
from myspellchecker.commands.train import _cmd_train_model

__all__ = [
    "_cmd_build",
    "_cmd_check",
    "_cmd_completion",
    "_cmd_config",
    "_cmd_infer_pos",
    "_cmd_segment",
    "_cmd_train_model",
    "validate_build_inputs",
]
