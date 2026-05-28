#include <torch/extension.h>

namespace {
bool g_comm_initialized = false;
}

void comm_init()
{
    TORCH_CHECK(!g_comm_initialized, "async_comm is already initialized");
    g_comm_initialized = true;
}

void comm_finalize()
{
    g_comm_initialized = false;
}

bool comm_is_initialized()
{
    return g_comm_initialized;
}

TORCH_LIBRARY(async_comm, m) {
    m.def("init() -> ()");
    m.def("finalize() -> ()");
    m.def("is_initialized() -> bool");
}

TORCH_LIBRARY_IMPL(async_comm, CPU, m) {
    m.impl("init", &comm_init);
    m.impl("finalize", &comm_finalize);
    m.impl("is_initialized", &comm_is_initialized);
}
