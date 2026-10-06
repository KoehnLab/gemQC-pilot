#include "gemqc/resources.h"

#include <cstdio>

#ifdef GEMQC_HAVE_OPENMP
#include <omp.h>
#endif

namespace gemqc::resources {

namespace {
size_t memory_ = 0;
int threads_ = 0;
}  // namespace

void set_memory(size_t bytes) { memory_ = bytes; }
size_t memory() { return memory_; }

void set_threads(int n) { threads_ = n > 0 ? n : 0; }

int threads() {
#ifdef GEMQC_HAVE_OPENMP
  return threads_ > 0 ? threads_ : omp_get_max_threads();
#else
  return 1;
#endif
}

bool fits(size_t bytes) { return memory_ == 0 || bytes <= memory_; }

std::string format_bytes(size_t bytes) {
  const char* unit[] = {"B", "kB", "MB", "GB", "TB"};
  double v = double(bytes);
  int i = 0;
  while (v >= 1024.0 && i < 4) v /= 1024.0, ++i;
  char buf[32];
  std::snprintf(buf, sizeof buf, i ? "%.2f %s" : "%.0f %s", v, unit[i]);
  return buf;
}

}  // namespace gemqc::resources
