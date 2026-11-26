"""Start the program and handle unexpected errors."""

import sys

from .app import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except MemoryError:
        print("Critical error: out of memory.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
