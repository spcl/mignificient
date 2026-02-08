
import importlib.machinery
import importlib.util
import json
import numpy as np
import os

import mignificient

if __name__ == "__main__":

    function_file = os.environ["FUNCTION_FILE"]
    function_name = os.environ["FUNCTION_HANDLER"]
    container_name = os.environ["CONTAINER_NAME"]

    ipc_backend = os.environ["IPC_BACKEND"]
    if ipc_backend == "iceoryx1":
        ipc_backend = mignificient.IPCBackend.ICEORYX_V1
    elif ipc_backend == "iceoryx2":
        ipc_backend = mignificient.IPCBackend.ICEORYX_V2
    else:
        raise RuntimeError(f"Unknown IPC backend: {ipc_backend}")

    func = None

    runtime = mignificient.Runtime(ipc_backend, container_name)
    runtime.register_runtime()

    while True:

        if func is None:

            name = os.path.basename(function_file)
            loader = importlib.machinery.SourceFileLoader(name, function_file)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            assert spec
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            func = getattr(mod, function_name)

        invocation_data = runtime.loop_wait()
        if invocation_data.size == 0:
            print("Empty payload, quit")
            break

        size = func(mignificient.Invocation(runtime, invocation_data, runtime.result()))

        runtime.finish(size)

