"""HQ child-process lifecycle; no model downloads or modifications."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import URLError

from .checkpoint import CheckpointError, validate_checkpoint, validate_dflash_helper


class ConfigurationError(ValueError):
    pass


def positive_seconds(value, name):
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite positive number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ConfigurationError(f"{name} must be a finite positive number of seconds")
    return seconds


def launch_configuration(root: Path, model: Path, environment: dict[str, str]):
    root, model = root.resolve(), model.resolve()
    if not model.is_dir():
        raise ConfigurationError(f"model directory does not exist: {model}")
    try:
        validate_checkpoint(model, environment)
    except CheckpointError as exc:
        raise ConfigurationError(str(exc)) from exc
    for word in shlex.split(environment.get("EXTRA_ARGS", "")):
        flag = word.split("=", 1)[0].replace("_", "-")
        if flag in ("--host", "--port", "--api-key", "--served-model-name", "--uds", "--config"):
            raise ConfigurationError(f"EXTRA_ARGS cannot override adapter-owned {flag}")
    profile = environment.get("HQ_PROFILE", "single")
    scripts = {"single": "single-user/start_qwen.sh", "batch": "batch/start_qwen.sh"}
    if profile not in scripts:
        raise ConfigurationError("HQ_PROFILE must be single or batch")
    script = root / scripts[profile]
    if not script.is_file():
        raise ConfigurationError(f"missing HQ launcher: {script}")
    # Never invoke docker/entrypoint.sh: it prepares/downloads default weights.
    env = dict(environment)
    if (profile == 'single' and env.get('SPEC') == 'dflash2'
            and (env.get('DRAFT_METHOD') or 'dflash') == 'dflash' and env.get('DRAFT')):
        helper = Path(env['DRAFT'])
        try:
            if not helper.is_absolute():
                helper = (model / helper).resolve()
                if not helper.is_relative_to(model):
                    raise ValueError('relative DRAFT directory escapes the model directory')
            else:
                helper = helper.resolve()
            validate_dflash_helper(helper)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConfigurationError(f'invalid DFlash helper: {exc}') from exc
        env['DRAFT'] = str(helper)
    env.pop("HQ_PROFILE", None)
    if "HQ_NATIVE_PYTHONPATH" in env:
        env["PYTHONPATH"] = env["HQ_NATIVE_PYTHONPATH"]
    env["MODEL"] = str(model)
    env["HOST"] = "127.0.0.1"
    # PRINT_ARGV would turn a successful spawn into a non-serving process.
    env.pop("PRINT_ARGV", None)
    return ["bash", str(script)], env


class Runtime:
    def __init__(self):
        self.process: subprocess.Popen | None = None

    def start(self, command, environment, cwd):
        if self.process is not None:
            raise RuntimeError("runtime already started; unload before reconfiguration")
        guardian = [sys.executable, str(Path(__file__).with_name('guardian.py')), str(os.getpid())]
        self.process = subprocess.Popen(guardian + list(command), cwd=cwd, env=environment, start_new_session=True)

    def wait_ready(self, base_url, api_key, timeout=600):
        deadline = time.monotonic() + timeout
        opener = build_opener(ProxyHandler({}))
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("HQ exited before readiness; inspect backend stderr")
            try:
                request = Request(base_url + "/v1/models", headers={"Authorization": f"Bearer {api_key}"})
                with opener.open(request, timeout=2) as response:
                    if response.status == 200:
                        document = json.load(response)
                        if any(model.get("id") == "qwen3.8-27b" for model in document.get("data", [])):
                            return
            except (URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError):
                pass
            time.sleep(0.25)
        raise TimeoutError("HQ model readiness timed out")

    def stop(self, timeout=10):
        process = self.process
        if process is None:
            return
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Reap workers even if the launcher exited before its descendants.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            self.process = None
