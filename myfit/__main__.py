# -*- coding: utf-8 -*-
# ruff: noqa: D401
"""Entry point."""
import click

from . import __version__


def _main() -> None:
    """Run main function for entrypoint."""

    @click.group(chain=True)
    @click.version_option(__version__)
    def entry_point() -> None:
        """Package entry point."""

    # entry_point.add_command(command)

    entry_point()


if __name__ == "__main__":
    _main()
