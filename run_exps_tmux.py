#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
from typing import List, Tuple

def ensure_tmux_available() -> None:
    try:
        subprocess.run(["tmux", "-V"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("Error: 'tmux' is not available on PATH.")

def read_commands(file_path: str) -> List[str]:
    if not os.path.isfile(file_path):
        sys.exit(f"Error: commands file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#") and line.strip().startswith("cd")]

def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    text = re.sub(r"-+", "-", text).strip("-") or "win"
    return text[:max_len]

def derive_window_name(cmd: str, fallback_index: int) -> str:
    match = re.search(r"--config\s+(\S+)", cmd)
    if match:
        base = os.path.splitext(os.path.basename(match.group(1)))[0]
        return slugify(base)
    return f"w-{fallback_index:02d}"

def session_exists(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def kill_session(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def tmux_new_session(session: str, window_name: str, command: str) -> None:
    cmd = f"bash -lic {shlex.quote(command)}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", window_name, cmd], check=True)

def tmux_new_window(session: str, window_name: str, command: str) -> None:
    cmd = f"bash -lic {shlex.quote(command)}"
    subprocess.run(["tmux", "new-window", "-t", session, "-n", window_name, "-k", cmd], check=True)

def tmux_attach(session: str) -> None:
    subprocess.run(["tmux", "attach-session", "-t", session])

def plan(session: str, commands: List[str]) -> List[Tuple[str, str]]:
    result, used = [], set()
    cnt = 0
    for i, cmd in enumerate(commands):
        # print(i, cmd)
        # if "_fashionmnist_" not in cmd and "_svhn_" not in cmd:
        # if "atkf_-1" in cmd:
        result.append((f'w-{cnt:02d}', cmd))
        cnt += 1
    return result

def wrap_command(cmd: str, prelude: str = None, show_cmd: bool = False, stay_open: bool = False) -> str:
    if prelude:
        cmd = f"{prelude} ; {cmd}"
    if show_cmd:
        # cmd = f"printf '$ %s\\n' {shlex.quote(cmd)} ; {cmd}"
        cmd = f"printf '$ %s\\n' {shlex.quote(cmd)} ; echo ; {cmd}"
    if stay_open:
        cmd = f'{cmd} ; echo; echo "==== DONE at $(date) ====" ; exec bash -li'
    return cmd

def main() -> None:
    parser = argparse.ArgumentParser(description="Run commands from a file in a tmux session (one window per command).")
    parser.add_argument("--commands-file", default="./commands.txt")
    parser.add_argument("--session", default="fl-experiments")
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stay-open", action="store_true")
    parser.add_argument("--prelude", default=None)
    parser.add_argument("--show-cmd", action="store_true")
    args = parser.parse_args()

    ensure_tmux_available()

    commands = read_commands(args.commands_file)
    if not commands:
        return

    windows = plan(args.session, commands)

    print(f"Session: {args.session}")
    print("Windows to create:\n")
    for name, cmd in windows:
        print(f" - {name}: {cmd}")

    if args.dry_run:
        return

    if session_exists(args.session):
        if args.force:
            print(f"Killing existing session '{args.session}'...")
            kill_session(args.session)
        else:
            sys.exit(f"Error: session '{args.session}' already exists. Use --force to replace.")

    first_name, first_cmd = windows[0]
    print(f"Creating session '{args.session}' with \n\t First window '{first_name}'...")
    tmux_new_session(args.session, first_name, wrap_command(first_cmd, args.prelude, args.show_cmd, args.stay_open))

    for name, cmd in windows[1:]:
        print(f"Creating window '{name}'...")
        tmux_new_window(args.session, name, wrap_command(cmd, args.prelude, args.show_cmd, args.stay_open))
        # time.sleep(1)

    if args.attach:
        tmux_attach(args.session)

if __name__ == "__main__":
    main()

# python run_exps_tmux.py --commands-file command-exps.txt --session n2 --force --attach --show-cmd --stay-open