/**
 * @file delta_bar_delta.hpp
 * @author Ranjodh Singh
 *
 * Implementation of DeltaBarDelta class wrapper.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DELTA_BAR_DELTA_IMPL_HPP
#define ENSMALLEN_DELTA_BAR_DELTA_IMPL_HPP

// In case it hasn't been included yet.
#include "./delta_bar_delta.hpp"

namespace ens {

inline DeltaBarDelta::DeltaBarDelta(
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const double kappa,
    const double phi,
    const double theta,
    const double minStepSize,
    const bool resetPolicy) : 
    optimizer(stepSize,
              maxIterations,
              tolerance,
              DeltaBarDeltaUpdate(stepSize, kappa, phi, theta, minStepSize),
              NoDecay(),
              resetPolicy)
{
  /* Nothing to do. */
}

} // namespace ens

#endif // ENSMALLEN_DELTA_BAR_DELTA_IMPL_HPP
