#include "ceres_cxx/include/ceres_example.h"
#include "ceres_cxx/src/main.rs.h"
// #include <algorithm>
#include <ceres/ceres.h>
// #include <functional>
// #include <glog/logging.h>
#include <iostream>
// #include <set>
#include <string>
// #include <unordered_map>

namespace org {
namespace ceres_example {

// Toy implementation of an in-memory ceres_example.
//
class CeresExample::impl {
  friend CeresExample;

  using CostFunctor = struct {
    template <typename T>
      bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
      }
  };

  // TODO(lucasw) not sure how to do autodiff with rust yet, so try numeric diff first
  using RustCostFunctor = struct {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
      // residual[0] = 10.0 - x[0];
      // residual[0] = evaluate<T>(x[0]);

      // When trying to use this with AutoDiff:
      // "cargo:warning=src/ceres_example.cc:34:31: error: cannot convert ‘const ceres::Jet<double, 1>’ to ‘double’"
      // So would have to implement Jet in rust to make that work?  Or could expose the Jet C++ type to rust?
      // or could expose all Jet math operations be exposed through a custom api- but then don't have much
      // commonality on the rust side between using Jet and double- the whole point is to be able to
      // write a native rust function once with regular math operations and then use it both in pure rust
      // and for ceres.

      // TODO(lucasw) need to detect that T is a C++ Jet, then change it into simpler type
      // that is just a plain double for the Scalar and array of doubles for the infinitesimal part
      // or can overload Jet below
      residual[0] = evaluate(x[0]);
      return true;
    }

    // template <typename T, int N>
    // bool operator()(const ceres::Jet<T, N>* const x, ceres::Jet<T, N>* residual) const {
    template <int N>
    bool operator()(const ceres::Jet<double, N>* const x, ceres::Jet<double, N>* residual) const {
      // residual[0].a = evaluate(x[0].a);
      // evaluate_raw_jet(N, x[0].a, x[0].v.data(), residual[0].a, residual[0].v.data());
      // rust::Slice<const double> x_v{x[0].v.data(), N};
      const std::array<double, N> x_v = *reinterpret_cast<const std::array<double, N>*>(x[0].v.data());
      // rust::Slice<double> residual_v{residual[0].v.data(), N};
      std::array<double, N>* residual_v = reinterpret_cast<std::array<double, N>*>(residual[0].v.data());
      evaluate_raw_jet(x[0].a, x_v, residual[0].a, *residual_v);
      std::cout << "C++ val " << x[0] << " " << residual[0] << ", length " << N << "\n";
      return true;
    }

  };

};

CeresExample::CeresExample() : impl(new class CeresExample::impl) {
  // TODO(lucasw) what happens if no google logging?
  // const char* test = "test";
  // google::InitGoogleLogging(test);
}

// TODO(lucasw) move this into impl?
void CeresExample::run(const rust::Vec<double>& vals) const {
// void CeresExample::run(const rust::Vec<T>& vals) const {
  // The variable to solve for with its initial value.
  // TODO(lucasw) pass in initial_x

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  /*
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CeresExample::impl::CostFunctor, 1, 1>(
          new CeresExample::impl::CostFunctor);
*/

  // presumably this is a little slower than autodifferentation for some problems


  // Build the problem.
  ceres::Problem problem;
  {
    double x = vals[0];
    std::cout << "\nnumeric diff\n";
    ceres::CostFunction* cost_function =
        new ceres::NumericDiffCostFunction<CeresExample::impl::RustCostFunctor, ceres::FORWARD, 1, 1>(
            new CeresExample::impl::RustCostFunctor);

    problem.AddResidualBlock(cost_function, nullptr, &x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << vals[0]
              << " -> " << x << "\n";
  }

  {
    double x = vals[0];
    std::cout << "\nauto diff\n";
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<CeresExample::impl::RustCostFunctor, 1, 1>(
            new CeresExample::impl::RustCostFunctor);

    problem.AddResidualBlock(cost_function, nullptr, &x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << vals[0]
              << " -> " << x << "\n";
  }
}

std::unique_ptr<CeresExample> new_ceres_example() {
  return std::make_unique<CeresExample>();
}

} // namespace ceres_example
} // namespace org
