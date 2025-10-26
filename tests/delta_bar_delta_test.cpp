/**
 * @file delta_bar_delta_test.cpp
 * @author Ranjodh Singh
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("DeltaBarDelta_GDTestFunction", "[DeltaBarDelta]",
   ENS_ALL_TEST_TYPES)
{
  DeltaBarDelta s(0.01, 500, 1e-9, 0.001, 0.2, 0.8);
  FunctionTest<GDTestFunction, TestType>(s,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("DeltaBarDelta_RosenbrockFunction", "[DeltaBarDelta]",
    ENS_ALL_TEST_TYPES)
{
  DeltaBarDelta s(0.001, 0, Tolerances<TestType>::Obj / 100,
      0.0001, 0.2, 0.8);
  FunctionTest<RosenbrockFunction, TestType>(s,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("DeltaBarDelta_LogisticRegressionFunction",
    "[DeltaBarDelta]", ENS_ALL_TEST_TYPES)
{
  DeltaBarDelta s(0.00032, 32, Tolerances<TestType>::Obj,
      0.000032, 0.2, 0.8);
  LogisticRegressionFunctionTest<TestType>(s);
}
