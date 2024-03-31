from ._mignificient import *

class Invocation:

    def __init__(self, runtime: _mignificient.Runtime, payload: _mignificient.Payload, result: _mignificient.Result):

        self.runtime = runtime
        self.payload = payload
        self.result = result

    def gpu_yield(self):

        self.runtime.gpu_yield()

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
