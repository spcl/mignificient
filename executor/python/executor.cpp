
#include "executor.hpp"

#if defined(WITH_PYTHON)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void define_mignificient_runtime(py::module& m)
{

  m.doc() = "mignificient executor module";

  // FIXME: function
  m.attr("__name__") = "_mignificient.function";

  py::class_<mignificient::executor::Runtime>(m, "Runtime")
      .def(py::init<const std::string&>())
      .def("loop_wait", &mignificient::executor::Runtime::loop_wait)
      .def("gpu_yield", &mignificient::executor::Runtime::gpu_yield)
      .def("finish", &mignificient::executor::Runtime::finish);

  py::class_<mignificient::executor::InvocationData>(m, "Buffer")
      .def(py::init())
      .def_readonly("data", &mignificient::executor::InvocationData::data)
      .def_readonly("size", &mignificient::executor::InvocationData::size)
      .def(
          "view_readable",
          [](mignificient::executor::InvocationData& self) {
            return py::memoryview::from_memory(self.data, sizeof(std::byte) * self.size);
          }
      );
}

PYBIND11_MODULE(_mignificient, m)
{

  define_mignificient_runtime(m);
}

#endif
