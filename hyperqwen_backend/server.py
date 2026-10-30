"""LocalAI gRPC bridge supervising a native HQ HTTP server."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import signal
import socket

import grpc
import httpx
import backend_pb2 as pb
import backend_pb2_grpc as rpc

from hyperqwen_backend.runtime import Runtime, launch_configuration, positive_seconds
from hyperqwen_backend.transport import prediction_payload, reply_from_json
from hyperqwen_backend.load_settings import apply_load_settings


class Backend(rpc.BackendServicer):
    def __init__(self):
        self.runtime = Runtime()
        self.identity = None
        self.client = None
        self.lock = asyncio.Lock()
        self.active = set()

    async def Health(self, request, context):
        if self.identity is not None and (self.runtime.process is None or self.runtime.process.poll() is not None):
            await context.abort(grpc.StatusCode.UNAVAILABLE, "HQ runtime exited")
        return pb.Reply(message=b"OK")

    async def LoadModel(self, request, context):
        async with self.lock:
            if self.identity is not None:
                return pb.Result(success=False, message="already loaded; unload before reconfiguration")
            try:
                if request.EngineArgs or request.Options:
                    raise ValueError("use HQ startup env, not engine_args/options")
                environment = apply_load_settings(request, dict(os.environ))
                startup_timeout = positive_seconds(os.environ.get("HQ_STARTUP_TIMEOUT", "600"), "HQ_STARTUP_TIMEOUT")
                request_timeout = positive_seconds(os.environ.get("HQ_REQUEST_TIMEOUT", "600"), "HQ_REQUEST_TIMEOUT")
                root = Path(os.environ.get("HQ_ROOT", "/app"))
                model = Path(request.Model)
                if not model.is_absolute() and request.ModelPath:
                    base = Path(request.ModelPath).resolve()
                    model = (base / model).resolve()
                    if not model.is_relative_to(base):
                        raise ValueError("model directory escapes LocalAI models path")
                command, env = launch_configuration(root, model, environment)
                # Reserve a local port briefly. HQ cannot inherit a listening fd;
                # a competing bind causes a clear startup failure, never reuse it.
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                self.key = secrets.token_hex(32)
                env.update(PORT=str(port), VLLM_API_KEY=self.key)
                base_url = f"http://127.0.0.1:{port}"
                self.runtime.start(command, env, root)
                await asyncio.to_thread(self.runtime.wait_ready, base_url, self.key,
                                        startup_timeout)
                self.client = httpx.AsyncClient(base_url=base_url, headers={"Authorization": "Bearer " + self.key},
                                                timeout=httpx.Timeout(request_timeout, connect=min(10, request_timeout)), trust_env=False)
                self.identity = request.Model
                return pb.Result(success=True, message="HyperQwen ready")
            except asyncio.CancelledError:
                await asyncio.to_thread(self.runtime.stop)
                raise
            except Exception as exc:
                await asyncio.to_thread(self.runtime.stop)
                return pb.Result(success=False, message=str(exc))

    async def Free(self, request, context):
        async with self.lock:
            self.identity = None
            pending = list(self.active)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if self.client:
                await self.client.aclose()
                self.client = None
            await asyncio.to_thread(self.runtime.stop)
        return pb.Result(success=True)

    async def _generate(self, request, context, streaming):
        if self.identity is None or self.client is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model not loaded")
        if request.ModelIdentity and request.ModelIdentity != self.identity:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model identity mismatch")
        task = asyncio.current_task()
        self.active.add(task)
        try:
            try:
                path, payload = prediction_payload(request, "qwen3.8-27b", streaming)
            except (ValueError, TypeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            # Closing the async response on cancellation closes upstream HTTP.
            async with self.client.stream("POST", path, json=payload) as response:
                response.raise_for_status()
                if not streaming:
                    yield reply_from_json(json.loads(await response.aread()), pb)
                else:
                    data = []
                    done = False
                    async for line in response.aiter_lines():
                        if line == "":
                            if data:
                                value = "\n".join(data)
                                data = []
                                if value == "[DONE]":
                                    done = True
                                    break
                                yield reply_from_json(json.loads(value), pb, True)
                        elif line.startswith("data:"):
                            data.append(line[5:].removeprefix(" "))
                    if not done:
                        raise ValueError("HQ stream ended without completion marker")
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            await context.abort(grpc.StatusCode.DATA_LOSS, "invalid HQ response")
        except httpx.TimeoutException:
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "HQ HTTP request timed out")
        except httpx.HTTPError:
            # Do not expose upstream request bodies or internal credentials.
            await context.abort(grpc.StatusCode.UNAVAILABLE, "HQ HTTP request failed; inspect runtime logs")
        finally:
            self.active.discard(task)

    async def Predict(self, request, context):
        async for reply in self._generate(request, context, False):
            return reply

    async def PredictStream(self, request, context):
        async for reply in self._generate(request, context, True):
            yield reply


def create_server(backend):
    value = os.environ.get("HQ_GRPC_MAX_MESSAGE_MB", "50")
    try:
        megabytes = int(value)
    except ValueError as exc:
        raise ValueError("HQ_GRPC_MAX_MESSAGE_MB must be an integer from 1 to 512") from exc
    if not 1 <= megabytes <= 512:
        raise ValueError("HQ_GRPC_MAX_MESSAGE_MB must be an integer from 1 to 512")
    limit = megabytes * 1024 * 1024
    server = grpc.aio.server(options=[('grpc.max_send_message_length', limit),
                                      ('grpc.max_receive_message_length', limit)])
    rpc.add_BackendServicer_to_server(backend, server)
    return server


async def watch_parent(parent, stop):
    """A managed backend must stop when its controller exits without Free/SIGTERM."""
    while not stop.is_set():
        if os.getppid() != parent:
            stop.set()
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass


async def serve(address):
    parent = os.getppid()
    backend = Backend()
    server = create_server(backend)
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"cannot bind gRPC address {address}")
    await server.start()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    monitor = asyncio.create_task(watch_parent(parent, stop))
    try:
        await stop.wait()
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        await server.stop(5)
        await backend.Free(None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1:50051")
    args = parser.parse_args()
    asyncio.run(serve(args.addr))
