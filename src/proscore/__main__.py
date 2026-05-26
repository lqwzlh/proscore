"""ProScore CLI entry point.

Usage::

    proscore run pipeline.xlsx [--output-script script.py]
    proscore template [output_dir]
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == "run":
        _cmd_run(args[1:])
    elif cmd == "template":
        _cmd_template(args[1:])
    elif cmd in ("-h", "--help"):
        print(__doc__)
    else:
        print(f"Unknown command: {cmd!r}")
        print(__doc__)
        sys.exit(1)


def _cmd_run(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a pipeline from Excel config")
    parser.add_argument("config", help="Path to pipeline.xlsx")
    parser.add_argument("--output-script", "-o", default=None,
                        help="Also generate a self-contained Python script")
    opts = parser.parse_args(args)

    from proscore._pipeline_config import run_pipeline

    try:
        run_pipeline(opts.config, output_script=opts.output_script)
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _cmd_template(args: list[str]) -> None:
    out_dir = args[0] if args else "."
    from proscore._pipeline_config import generate_template

    path = generate_template(out_dir)
    print(f"模板已生成: {path}")


if __name__ == "__main__":
    main()
