#include "gemqc/info.h"

#include "gemqc/basis.h"

#include <libint2/config.h>

namespace gemqc {

std::map<std::string, int> libint_info() {
  return {
      {"version_major", LIBINT_MAJOR_VERSION},
      {"version_minor", LIBINT_MINOR_VERSION},
      {"version_micro", LIBINT_MICRO_VERSION},
      {"max_am", LIBINT_MAX_AM},
      {"max_am_auxiliary", max_l_auxiliary()},
      {"onebody_max_deriv", LIBINT_INCLUDE_ONEBODY},  // -1: not included
      {"eri_max_deriv", LIBINT_INCLUDE_ERI},
      {"eri2_max_deriv", LIBINT_INCLUDE_ERI2},
      {"eri3_max_deriv", LIBINT_INCLUDE_ERI3},
  };
}

}  // namespace gemqc
