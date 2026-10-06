#include "gemqc/log.h"

#include <cstdarg>
#include <cstdio>
#include <iostream>

namespace gemqc::log {

namespace {
Sink sink_;
Flush flush_;
bool enabled_ = true;
}  // namespace

void set_sink(Sink sink, Flush flush) {
  sink_ = std::move(sink);
  flush_ = std::move(flush);
}
void set_enabled(bool on) { enabled_ = on; }
bool enabled() { return enabled_; }

void line(const std::string& text) {
  if (!enabled_) return;
  if (sink_)
    sink_(text);
  else
    std::cout << text << '\n';
}

void flush() {
  if (!enabled_) return;
  if (sink_) {
    if (flush_) flush_();
  } else {
    std::cout.flush();
  }
}

std::string format(const char* fmt, ...) {
  va_list args, args2;
  va_start(args, fmt);
  va_copy(args2, args);
  const int n = std::vsnprintf(nullptr, 0, fmt, args);
  va_end(args);
  std::string s(n > 0 ? n : 0, '\0');
  std::vsnprintf(s.data(), s.size() + 1, fmt, args2);
  va_end(args2);
  return s;
}

}  // namespace gemqc::log
