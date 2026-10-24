import unittest
import backend_pb2 as pb
from hyperqwen_backend.transport import prediction_payload, reply_from_json


class TransportTests(unittest.TestCase):
    def test_zero_temperature(self):
        path, payload = prediction_payload(pb.PredictOptions(Prompt="Hi", Temperature=0), "model")
        self.assertEqual(path, "/v1/completions")
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("top_p", payload)

    def test_chat_tools_and_template(self):
        request = pb.PredictOptions(Tools='[{"type":"function"}]', ToolChoice='"auto"', Metadata={"enable_thinking": "false"})
        request.Messages.add(role="assistant", content="", tool_calls='[{"id":"call1"}]')
        request.Messages.add(role="tool", content="result", tool_call_id="call1")
        path, payload = prediction_payload(request, "model", True)
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["messages"][1]["tool_call_id"], "call1")
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["tool_choice"], "auto")

    def test_reject_ambiguous_prompt(self):
        request = pb.PredictOptions(Prompt="rendered")
        request.Messages.add(role="user", content="Hi")
        with self.assertRaises(ValueError):
            prediction_payload(request, "model")

    def test_structured_reply(self):
        document = {"choices": [{"delta": {"content": "hello", "reasoning_content": "think", "tool_calls": [{"index": 2, "id": "call1", "function": {"name": "lookup", "arguments": "{"}}]}}], "usage": {"completion_tokens": 4, "prompt_tokens": 8}}
        reply = reply_from_json(document, pb, True)
        self.assertEqual(reply.chat_deltas[0].tool_calls[0].index, 2)
        self.assertEqual(reply.chat_deltas[0].reasoning_content, "think")
        self.assertEqual(reply.tokens, 4)
        self.assertEqual(reply.prompt_tokens, 8)

    def test_sampler_and_logprob_mapping(self):
        request = pb.PredictOptions(Prompt='Hi', Penalty=1.1, LogitBias='{"12":-2}', Logprobs=3)
        _, payload = prediction_payload(request, 'model')
        self.assertAlmostEqual(payload['repetition_penalty'], 1.1, places=5)
        self.assertEqual(payload['logit_bias'], {'12': -2})
        self.assertEqual(payload['logprobs'], 3)
        request.Prompt = ''
        request.Messages.add(role='user', content='Hi')
        request.Metadata['reasoning_effort'] = 'high'
        _, payload = prediction_payload(request, 'model')
        self.assertTrue(payload['logprobs'])
        self.assertEqual(payload['top_logprobs'], 3)
        self.assertEqual(payload['reasoning_effort'], 'high')

    def test_invalid_structured_options(self):
        for request in (pb.PredictOptions(LogitBias='[]'), pb.PredictOptions(Grammar='gbnf'),
                        pb.PredictOptions(Mirostat=1), pb.PredictOptions(Tokens=-1)):
            with self.assertRaises(ValueError):
                prediction_payload(request, 'model')
        request = pb.PredictOptions(Tools='{}')
        request.Messages.add(role='user', content='Hi')
        with self.assertRaises(ValueError):
            prediction_payload(request, 'model')

    def test_invalid_numeric_values(self):
        requests = [pb.PredictOptions(Temperature=float('nan')), pb.PredictOptions(TopP=1.5),
                    pb.PredictOptions(MinP=-1), pb.PredictOptions(TopK=-2),
                    pb.PredictOptions(Logprobs=-1), pb.PredictOptions(Penalty=-1),
                    pb.PredictOptions(LogitBias='{"1":NaN}'), pb.PredictOptions(LogitBias='{"1":true}'),
                    pb.PredictOptions(LogitBias='{"1":' + '9' * 1000 + '}')]
        for request in requests:
            with self.assertRaises(ValueError):
                prediction_payload(request, 'model')

    def test_usage_only(self):
        reply = reply_from_json({"choices": [], "usage": {"completion_tokens": 12}}, pb, True)
        self.assertEqual(reply.tokens, 12)
        self.assertFalse(reply.chat_deltas)


if __name__ == "__main__":
    unittest.main()
