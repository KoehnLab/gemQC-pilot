#pragma once
/// \file
/// Output channel of the C++ core. Lines go to a sink function (set from
/// Python so that output is ordered with Python's own printout), else stdout.

#include <functional>
#include <string>

namespace gemqc::log {

using Sink = std::function<void(const std::string&)>;
using Flush = std::function<void()>;

void set_sink(Sink sink, Flush flush = {});  ///< empty sink: back to stdout
void set_enabled(bool on);          ///< switch all output on/off
bool enabled();
void line(const std::string& text = "");  ///< print one line
void flush();  ///< flush output (after progress lines, e.g. per SCF iteration)

/// printf-style formatting into std::string
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

}  // namespace gemqc::log
