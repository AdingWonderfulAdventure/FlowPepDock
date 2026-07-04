import argparse
import shlex
import subprocess
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分")
    parts.append(f"{secs}秒")
    return "".join(parts)


def _format_command(cmd):
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    if len(cmd_str) > 200:
        return cmd_str[:197] + "..."
    return cmd_str


def _send_bark(bark_id, title, body, timeout=10):
    url = f"https://api.day.app/{bark_id}/{quote(title)}/{quote(body)}"
    with urlopen(url, timeout=timeout) as resp:
        resp.read()


def main():
    parser = argparse.ArgumentParser(
        description="Run a command and send Bark notification if it takes long enough."
    )
    parser.add_argument(
        "--bark_id",
        required=True,
        help="Bark device key.",
    )
    parser.add_argument(
        "--min_minutes",
        type=float,
        default=20.0,
        help="Notify only if elapsed minutes >= this value.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="FlowPepDock",
        help="Bark notification title.",
    )
    parser.add_argument(
        "--cmd",
        nargs=argparse.REMAINDER,
        help="Command to run (use: --cmd -- <cmd> <args...>).",
    )
    args = parser.parse_args()

    cmd = args.cmd or []
    if cmd[:1] == ["--"]:
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing --cmd; use: --cmd -- <cmd> <args...>")

    start = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - start

    if elapsed >= args.min_minutes * 60.0:
        cmd_display = _format_command(cmd)
        duration = _format_duration(elapsed)
        message = f"{cmd_display}已完成，耗时{duration}。"
        try:
            _send_bark(args.bark_id, args.title, message)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[WARN] Bark notify failed: {exc}", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
