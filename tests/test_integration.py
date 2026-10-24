import asyncio
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import grpc
import backend_pb2 as pb
import backend_pb2_grpc as rpc
from hyperqwen_backend.server import Backend, create_server
from tests.checkpoint_fixture import create_checkpoint


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        create_checkpoint(self.model)
        self.launcher = self.root / 'single-user/start_qwen.sh'
        self.launcher.parent.mkdir()
        fake = Path(__file__).with_name('fake_hq.py').resolve()
        self.command = f'exec {shlex.quote(sys.executable)} {shlex.quote(str(fake))}\n'
        self.launcher.write_text(self.command)
        self.marker = self.root / 'disconnected'
        self.environment = patch.dict(os.environ, {'HQ_ROOT': str(self.root), 'HQ_STARTUP_TIMEOUT': '5',
                                                   'HQ_FAKE_DISCONNECT_MARKER': str(self.marker)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.backend = Backend()
        self.server = create_server(self.backend)
        port = self.server.add_insecure_port('127.0.0.1:0')
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
        self.stub = rpc.BackendStub(self.channel)

    async def asyncTearDown(self):
        await self.channel.close()
        await self.server.stop(0)
        await self.backend.Free(None, None)

    async def load(self):
        result = await self.stub.LoadModel(pb.ModelOptions(Model=str(self.model)), timeout=10)
        self.assertTrue(result.success, result.message)

    def request(self, content='Hi'):
        request = pb.PredictOptions(ModelIdentity=str(self.model))
        request.Messages.add(role='user', content=content)
        return request

    async def test_load_concurrent_predict_stream_unload_reload(self):
        for _ in range(2):
            await self.load()
            process = self.backend.runtime.process
            replies = await asyncio.gather(*(self.stub.Predict(self.request()) for _ in range(4)))
            self.assertEqual([r.message for r in replies], [b'hello'] * 4)
            stream = [r async for r in self.stub.PredictStream(self.request())]
            self.assertEqual(stream[0].message, b'hello')
            with self.assertRaises(grpc.aio.AioRpcError):
                await self.stub.Predict(pb.PredictOptions(ModelIdentity='wrong'))
            self.assertTrue((await self.stub.Free(pb.HealthMessage())).success)
            self.assertIsNotNone(process.poll())
            self.assertIsNone(self.backend.runtime.process)

    async def test_cancellation_closes_upstream_and_keeps_runtime(self):
        await self.load()
        stream = self.stub.PredictStream(self.request('cancel'))
        self.assertEqual((await stream.read()).message, b'hello')
        stream.cancel()
        async with asyncio.timeout(5):
            while not self.marker.exists() or self.backend.active:
                await asyncio.sleep(0.02)
        self.assertIsNone(self.backend.runtime.process.poll())
        self.assertEqual((await self.stub.Predict(self.request())).message, b'hello')

    async def test_unload_cancels_live_stream(self):
        await self.load()
        process = self.backend.runtime.process
        stream = self.stub.PredictStream(self.request('cancel'))
        await stream.read()
        result = await self.stub.Free(pb.HealthMessage(), timeout=5)
        self.assertTrue(result.success)
        self.assertFalse(self.backend.active)
        self.assertIsNotNone(process.poll())
        stream.cancel()

    async def test_failed_start_can_retry(self):
        self.launcher.write_text('exit 7\n')
        result = await self.stub.LoadModel(pb.ModelOptions(Model=str(self.model)), timeout=10)
        self.assertFalse(result.success)
        self.assertIsNone(self.backend.runtime.process)
        self.launcher.write_text(self.command)
        await self.load()

    async def test_relative_model_path_and_identity(self):
        result = await self.stub.LoadModel(pb.ModelOptions(Model='model', ModelPath=str(self.root)))
        self.assertTrue(result.success, result.message)
        request = self.request()
        request.ModelIdentity = 'model'
        self.assertEqual((await self.stub.Predict(request)).message, b'hello')

    async def test_relative_path_escape_is_rejected(self):
        result = await self.stub.LoadModel(pb.ModelOptions(Model='../outside', ModelPath=str(self.root)))
        self.assertFalse(result.success)
        self.assertIn('escapes', result.message)
        self.assertIsNone(self.backend.runtime.process)

    async def test_invalid_timeout_fails_before_spawn(self):
        with patch.dict(os.environ, {'HQ_REQUEST_TIMEOUT': 'NaN'}):
            result = await self.stub.LoadModel(pb.ModelOptions(Model=str(self.model)))
        self.assertFalse(result.success)
        self.assertIn('HQ_REQUEST_TIMEOUT', result.message)
        self.assertIsNone(self.backend.runtime.process)

    async def test_load_deadline_cleans_starting_child(self):
        self.launcher.write_text('exec sleep 60\n')
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.stub.LoadModel(pb.ModelOptions(Model=str(self.model)), timeout=0.1)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        async with asyncio.timeout(5):
            while self.backend.runtime.process is not None:
                await asyncio.sleep(0.02)
        self.launcher.write_text(self.command)
        await self.load()

    async def test_request_above_default_grpc_four_megabytes(self):
        await self.load()
        reply = await self.stub.Predict(self.request('x' * (5 * 1024 * 1024)), timeout=10)
        self.assertEqual(reply.message, b'hello')

    async def test_invalid_grpc_limits(self):
        for value in ('0', '-1', '513', 'NaN', '1.5'):
            with patch.dict(os.environ, {'HQ_GRPC_MAX_MESSAGE_MB': value}):
                with self.assertRaises(ValueError):
                    create_server(Backend())

    async def test_dead_child_is_unhealthy(self):
        await self.load()
        self.backend.runtime.process.kill()
        await asyncio.to_thread(self.backend.runtime.process.wait)
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.stub.Health(pb.HealthMessage())
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAVAILABLE)
