import unittest
from unittest.mock import patch
import httpx
import grpc
import backend_pb2_grpc as rpc
import backend_pb2 as pb
from hyperqwen_backend.server import Backend


class Context:
    async def abort(self, code, message):
        raise RuntimeError(message)


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_predict_against_mock_http(self):
        def respond(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello", "reasoning_content": "reason"}}]})
        backend = Backend()
        backend.identity = "model"
        backend.client = httpx.AsyncClient(base_url="http://hq", transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(backend.client.aclose)
        request = pb.PredictOptions(ModelIdentity="model")
        request.Messages.add(role="user", content="Hi")
        reply = await backend.Predict(request, Context())
        self.assertEqual(reply.chat_deltas[0].content, "hello")
        self.assertEqual(reply.chat_deltas[0].reasoning_content, "reason")

    async def test_stream_against_mock_http(self):
        def respond(request):
            return httpx.Response(200, text=': keepalive\n\ndata: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n')
        backend = Backend()
        backend.identity = "model"
        backend.client = httpx.AsyncClient(base_url="http://hq", transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(backend.client.aclose)
        request = pb.PredictOptions()
        request.Messages.add(role="user", content="Hi")
        replies = [reply async for reply in backend.PredictStream(request, Context())]
        self.assertEqual(replies[0].message, b"hi")
        self.assertEqual(replies[1].tokens, 2)

    async def test_grpc_health_and_unloaded_prediction(self):
        server = grpc.aio.server()
        rpc.add_BackendServicer_to_server(Backend(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = rpc.BackendStub(channel)
                reply = await stub.Health(pb.HealthMessage())
                self.assertEqual(reply.message, b"OK")
                with self.assertRaises(grpc.aio.AioRpcError) as caught:
                    await stub.Predict(pb.PredictOptions(Prompt="Hi"))
                self.assertEqual(caught.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
        finally:
            await server.stop(0)

    async def test_typed_settings_reach_launcher(self):
        with patch('hyperqwen_backend.server.launch_configuration',
                   side_effect=ValueError('stop before spawn')) as launch:
            result = await Backend().LoadModel(
                pb.ModelOptions(Model='/unused', GPUMemoryUtilization=.5,
                                TensorParallelSize=2), Context())
        self.assertFalse(result.success)
        env = launch.call_args.args[2]
        self.assertEqual(env['GPU_UTIL'], '0.5')
        self.assertEqual(env['KV_MEM'], '')
        self.assertIn('--tensor-parallel-size 2', env['EXTRA_ARGS'])

    async def test_invalid_load_setting_is_rejected(self):
        result = await Backend().LoadModel(pb.ModelOptions(Model="/unused", GPUMemoryUtilization=1.5), Context())
        self.assertFalse(result.success)
        self.assertIn("GPUMemoryUtilization", result.message)

    async def test_malformed_upstream_is_data_loss(self):
        class CapturingContext:
            async def abort(self, code, message):
                self.code = code
                raise RuntimeError(message)
        backend = Backend()
        backend.identity = 'model'
        backend.client = httpx.AsyncClient(base_url='http://hq', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='not-json')))
        self.addAsyncCleanup(backend.client.aclose)
        context = CapturingContext()
        with self.assertRaisesRegex(RuntimeError, 'invalid HQ response'):
            await backend.Predict(pb.PredictOptions(Prompt='Hi'), context)
        self.assertEqual(context.code, grpc.StatusCode.DATA_LOSS)
        self.assertFalse(backend.active)

    async def test_truncated_stream_and_http_timeout(self):
        class CapturingContext:
            async def abort(self, code, message):
                self.code = code
                raise RuntimeError(message)
        def truncated(request):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        def timeout(request):
            raise httpx.ReadTimeout('fixture timeout')
        for handler, expected in ((truncated, grpc.StatusCode.DATA_LOSS),
                                  (timeout, grpc.StatusCode.DEADLINE_EXCEEDED)):
            backend = Backend()
            backend.identity = 'model'
            async with httpx.AsyncClient(base_url='http://hq', transport=httpx.MockTransport(handler)) as client:
                backend.client = client
                context = CapturingContext()
                with self.assertRaises(RuntimeError):
                    _ = [reply async for reply in backend.PredictStream(pb.PredictOptions(Prompt='Hi'), context)]
                self.assertEqual(context.code, expected)
                self.assertFalse(backend.active)

    async def test_unloaded_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            await Backend().Predict(pb.PredictOptions(), Context())
