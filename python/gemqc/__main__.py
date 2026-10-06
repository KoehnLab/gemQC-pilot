"""Command line entry point: gemqc input.yaml"""

import argparse
import sys

from .input import InputError


def main(argv=None):
    """Command line: gemqc [-q] [-m MEMORY] [-t THREADS] [-a PATH] input.yaml.

    Errors are printed on stdout and stderr; input errors exit with status 1.
    """
    parser = argparse.ArgumentParser(prog="gemqc", description="gemQC driver")
    parser.add_argument("input", help="YAML input file")
    parser.add_argument("-q", "--quiet", action="store_true", help="no screen output")
    parser.add_argument("-m", "--memory", help="memory limit, e.g. '4 GB' (overrides input)")
    parser.add_argument("-t", "--threads", type=int, help="number of threads (overrides input)")
    parser.add_argument("-a", "--archive", metavar="PATH",
                        help="path and base name of the results (.json), restart (.h5) and "
                             "cube files (default: those of the input file)")
    args = parser.parse_args(argv)

    from .driver import error_message, run
    from .input import parse_memory
    try:
        memory = parse_memory(args.memory) if args.memory else None
        run(args.input, quiet=args.quiet, memory=memory, threads=args.threads,
            archive=args.archive)
    except Exception as e:
        # error on stdout (if not yet done by run) and on stderr
        if not getattr(e, "gemqc_reported", False) and not args.quiet:
            print(error_message(e))
        sys.stdout.flush()
        if isinstance(e, (InputError, FileNotFoundError)):
            sys.exit(error_message(e))
        raise  # unexpected: message plus traceback on stderr


if __name__ == "__main__":
    main()
