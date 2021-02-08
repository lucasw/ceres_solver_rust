#pragma once
#include "rust/cxx.h"
#include <memory>
#include <vector>

namespace org {
namespace ceres_example {

struct WayPoint;

class CeresExample {
public:
  CeresExample();

  void run_numeric(const rust::Vec<double>& vals) const;
  void run_auto(const rust::Vec<double>& vals) const;

  void find_trajectory(const WayPoint& start, const WayPoint& target) const;
private:
  class impl;
  std::shared_ptr<impl> impl;
};

std::unique_ptr<CeresExample> new_ceres_example();

} // namespace ceres_example
} // namespace org
