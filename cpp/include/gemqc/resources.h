#pragma once
/// \file
/// Resource limits (main memory, threads) shared by all computational parts.

#include <cstddef>
#include <string>

namespace gemqc::resources {

void set_memory(size_t bytes);  ///< 0: unlimited
size_t memory();                ///< memory limit in bytes (0: unlimited)
void set_threads(int n);        ///< 0: OpenMP default
int threads();                  ///< number of threads to use (>= 1)

/// true if `bytes` fit into the memory limit
bool fits(size_t bytes);

/// human-readable size, e.g. "1.25 GB"
std::string format_bytes(size_t bytes);

}  // namespace gemqc::resources
