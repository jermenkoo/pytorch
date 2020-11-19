import os
import sys
import unittest

import torch
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import skipIfRocm, skipCUDANonDefaultStreamIf

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestCUDA(JitTestCase):
    """
    A suite of tests for the CUDA API in TorchScript.
    """
    @skipIfRocm
    def test_current_stream(self):
        @torch.jit.script
        def fn():
            device_index = torch.cuda.current_device()
            s0 = torch.cuda.current_stream(device_index)
            s1 = torch.cuda.current_stream(1)
            s2 = torch.cuda.current_stream(0)

            return s0.device_index(),  s1.device_index(), s2.device_index()

        s0, s1, s2 = fn()

        self.assertEqual(0, s0)
        self.assertEqual(1, s1)
        self.assertEqual(0, s2)
        self.assertEqual(s0, s2)

    @skipIfRocm
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self):
        @torch.jit.script
        def fn():
            s0 = torch.cuda.default_stream(0)
            s1 = torch.cuda.default_stream(1)

            d0 = torch.device('cuda:0')
            d1 = torch.device('cuda:1')

            with torch.cuda.device_jit(d0):
                s2 = torch.cuda.current_stream(0)

            with torch.cuda.device_jit(d1):
                s3 = torch.cuda.current_stream(1)

            check_s2 = s2.id() == s0.id()
            check_s3 = s3.id() == s1.id()
            return s0.device_index(), s1.device_index(), check_s2, check_s3

        s0, s1, check_s2, check_s3 = fn()
        self.assertEqual(s0, 0)
        self.assertEqual(s1, 1)
        self.assertEqual(check_s2, True)
        self.assertEqual(check_s3, True)

    @skipIfRocm
    def test_simple_stream(self):
        @torch.jit.script
        def fn():
            s = torch.classes.cuda.Stream(0, 0)
            return s is not None

        @torch.jit.script
        def fn1():
            device_index = torch.cuda.current_device()
            s = torch.classes.cuda.Stream(device_index, 0)
            return device_index == s.device_index()

        self.assertEqual(fn(), True, "Could not create Stream!")
        self.assertEqual(fn1(), True, "Could not create Stream!")


    @skipIfRocm
    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        @torch.jit.script
        def test_get_stream():
            device_index = torch.cuda.current_device()
            current_stream = torch.cuda.current_stream(device_index)
            default_stream = torch.cuda.default_stream(device_index)
            user_stream = torch.classes.cuda.Stream(device_index, 0)

            is_not_same = default_stream.id() != user_stream.id()

            with torch.cuda.stream(user_stream):
                is_stream_set = torch.cuda.current_stream(device_index).id() == user_stream.id()

            user_stream_query = user_stream.query()
            tensor1 = torch.rand(10000, 10000, device = "cuda")
            tensor2 = torch.mm(tensor1, tensor1)
            default_stream.synchronize()
            default_stream_query = default_stream.query()

            return is_not_same, is_stream_set, user_stream_query, default_stream_query, default_stream.id(), user_stream.id()

        is_not_same, is_stream_set, user_stream_query, default_stream_query, default_stream_id, user_stream_id = test_get_stream()

        self.assertEqual(is_not_same, True)
        self.assertEqual(is_stream_set, True)
        self.assertEqual(default_stream_id, 0)
        self.assertNotEqual(user_stream_id, 0)
        self.assertTrue(user_stream_query)
        self.assertTrue(default_stream_query)

    @skipIfRocm
    def test_stream_context(self):
        @torch.jit.script
        def fn():
            device_index = torch.cuda.current_device()
            user_stream = torch.classes.cuda.Stream(device_index, 0)
            A = torch.rand(1000, 1000, device = "cuda")

            with torch.cuda.stream(user_stream):
                check = torch.cuda.current_stream(device_index).id() == user_stream.id()
                B = torch.mm(A, A)
            return A, B, check

        A, B, is_stream_set = fn()
        self.assertEqual(torch.matmul(A, A), B)
        self.assertEqual(is_stream_set, True, "Error: Current stream was not set to user stream!")

        @torch.jit.script
        def test_multiple_stream():
            device_index = torch.cuda.current_device()
            s1 = torch.classes.cuda.Stream(device_index, 0)
            s2 = torch.classes.cuda.Stream(device_index, 0)

            A = torch.rand(1000, 1000, device = "cuda")
            B = torch.rand(1000, 1000, device = "cuda")
            with torch.cuda.stream(s1):
                C = torch.mm(A, A)
                is_stream_s1 = torch.cuda.current_stream(s1.device_index()).id() == s1.id()
                with torch.cuda.stream(s2):
                    is_stream_s2 = torch.cuda.current_stream(s2.device_index()).id() == s2.id()
                    D = torch.mm(B, B)
                # Wait for D to be computed
                s2.synchronize()
            return A, B, C, D, is_stream_s1, is_stream_s2

        A, B, C, D, is_stream_s1, is_stream_s2 = test_multiple_stream()
        self.assertEqual(torch.matmul(A, A), C)
        self.assertEqual(torch.matmul(B, B), D)
        self.assertEqual(is_stream_s1, True)
        self.assertEqual(is_stream_s2, True)

    @skipIfRocm
    def test_events(self):
        @torch.jit.script
        def test_simple_event():
            e = torch.classes.cuda.Event(True, False, False)
            return e is not None
        self.assertTrue(test_simple_event(), "Could not create CUDA Event!")

        @torch.jit.script
        def test_event_query() -> bool:
            s = torch.classes.cuda.Stream(0, 0)
            e = torch.classes.cuda.Event(True, False, False)
            e.record(s)
            return e.query()
        self.assertTrue(test_event_query())

        @torch.jit.script
        def test_event() -> float:
            device_index = torch.cuda.current_device()
            stream = torch.cuda.current_stream(device_index)
            event = torch.classes.cuda.Event(True, False, False)
            is_true_event_query = event.query()
            start_event = torch.classes.cuda.Event(True, False, False)
            stream.record_event(start_event)
            tensor1 = torch.rand(1000000000, 1000000000, device = "cuda")
            with torch.cuda.stream(stream):
                tensor2 = torch.mm(tensor1, tensor1)
            stream.record_event(event)
            event.synchronize()
            is_again_true_event_query = event.query()

            if not (is_true_event_query and is_again_true_event_query):
                return -1.0
            return start_event.elapsed_time(event)

        self.assertGreater(test_event(), 0)

        @torch.jit.script
        def test_stream_synchronize() -> float:
            device_index = torch.cuda.current_device()
            s = torch.classes.cuda.Stream(device_index, 0)
            e_tik = torch.classes.cuda.Event(True, False, False)
            e_tok = torch.classes.cuda.Event(True, False, False)

            e_tik.record(s)
            tensor1 = torch.rand(1000000000, 1000000000, device = "cuda")
            with torch.cuda.stream(s):
                tensor2 = torch.mm(tensor1, tensor1)
            e_tok.record(s)
            s.synchronize()

            if not s.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_stream_synchronize(), 0)

        @torch.jit.script
        def test_event_synchronize() -> float:
            device_index = torch.cuda.current_device()
            s = torch.classes.cuda.Stream(device_index, 0)
            e_tik = torch.classes.cuda.Event(True, False, False)
            e_tok = torch.classes.cuda.Event(True, False, False)

            e_tik.record(s)
            tensor1 = torch.rand(1000000000, 1000000000, device = "cuda")
            with torch.cuda.stream(s):
                tensor2 = torch.mm(tensor1, tensor1)

            s.record_event(e_tok)
            e_tok.synchronize()

            if not s.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_event_synchronize(), 0)

        @torch.jit.script
        def test_event_wait() -> float:
            device_index = torch.cuda.current_device()
            s0 = torch.cuda.current_stream(device_index)
            s1 = torch.classes.cuda.Stream(device_index, 0)
            e_tik = torch.classes.cuda.Event(True, True, False)
            e_tok = torch.classes.cuda.Event(True, True, False)

            e_tik.record(s0)
            tensor1 = torch.rand(1000000000, 1000000000, device = "cuda")
            with torch.cuda.stream(s0):
                tensor2 = torch.mm(tensor1, tensor1)
            e_sync = torch.classes.cuda.Event(True, False, False)
            e_sync.record(torch.cuda.current_stream(device_index))
            e_sync.wait(s1)
            with torch.cuda.stream(s1):
                tensor3 = torch.rand(1000000000, 1000000000, device = "cuda")
                tensor4 = torch.mm(tensor3, tensor3)
            s1.synchronize()
            e_tok.record(torch.cuda.current_stream(device_index))
            e_tok.synchronize()

            if not s0.query() or not s1.query() or not e_sync.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_event_wait(), 0)

        @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
        @torch.jit.script
        def test_events_wait():
            d0 = torch.device('cuda:0')
            d1 = torch.device('cuda:1')

            with torch.cuda.device_jit(d0):
                s0 = torch.cuda.current_stream(0)
                tensor1 = torch.rand(1000000000, 1000000000, device = "cuda")
                tensor2 = torch.mm(tensor1, tensor1)
                e0 = torch.classes.cuda.Event(False, False, False)
                s0.record_event(e0)

            with torch.cuda.device_jit(d1):
                s1 = torch.cuda.current_stream(1)

            s1.wait_event(e0)
            s1.synchronize()

            return e0.query() and s0.query() and s1.query()
        self.assertTrue(test_events_wait())

    @skipIfRocm
    def test_save_load(self):
        a = torch.ones(3, 4)
        b = torch.ones(3, 4)
        c = torch.cat((a, b), 0)

        class Model(torch.nn.Module):
            def forward(self):
                device_index = torch.cuda.current_device()
                s = torch.classes.cuda.Stream(device_index, 0)
                a = torch.ones(3, 4, device = "cuda")
                b = torch.ones(3, 4, device = "cuda")

                with torch.cuda.stream(s):
                    check = torch.cuda.current_stream(s.device_index()).id() == s.id()
                    c = torch.cat((a, b), 0)
                return check, c

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        check, output = script_model()
        # Check the output with the reference tensors declared earlier
        self.assertTrue(check)
        self.assertEqual(output.size(), c.size())
        script_model.save("saved_model.pt")

        # Check if the file was saved in CWD
        self.assertTrue(os.path.exists("saved_model.pt"))

        # Load the model and compare the output with the saved model
        load_model = torch.jit.load("saved_model.pt")
        check_load, output_load = load_model()
        self.assertTrue(check_load)
        self.assertEqual(output_load, output)
