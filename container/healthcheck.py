#!/usr/bin/env python3
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def main() -> int:
    try:
        with urlopen("http://127.0.0.1:8998/api/health", timeout=2) as response:
            return 0 if response.status == 200 else 1
    except (HTTPError, TimeoutError, URLError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
