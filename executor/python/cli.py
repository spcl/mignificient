
import json
import numpy as np
import os

import mignificient


class BufferWriter:

    def __init__(self, buf):
        self.buffer = buf
        self.view = buf.view_writable()

    def write(self, b):

        b_len = len(b)
        pos = self.buffer.size

        self.view[ pos : pos + b_len ] = b
        self.buffer.size = pos + b_len

class BufferStringWriter:

    def __init__(self, buf):
        self.buffer = buf
        self.view = buf.view_writable()

    def write(self, string):

        str_encoded = string.encode()
        str_len = len(str_encoded)
        pos = self.buffer.size

        self.view[ pos : pos + str_len] = str_encoded
        self.buffer.size = pos + str_len


if __name__ == "__main__":

    function_file = os.environ["FUNCTION_FILE"]
    function_name = os.environ["FUNCTION_NAME"]
    container_name = os.environ["CONTAINER_NAME"]

    runtime = mignificient.Runtime(container_name)

    while True:

        invocation_data = runtime.loop_wait()

        if invocation_data.size == 0:
            print("Empty payload, quit")
            break

        input_data = np.frombuffer(invocation_data.view_readable(), dtype=np.int32).astype(dtype=np.int_)
        print(f"Invoke, data size {invocation_data.size}, first element {input_data[0]}")

        res = "{ \"test\": 42 }"

        buf = runtime.result()
        json.dump(res, BufferStringWriter(buf))

        runtime.finish(buf.size)

