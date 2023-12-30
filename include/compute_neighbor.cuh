#pragma once
#include "fix-clangd-cuda-lint.hpp"
#include <Eigen/Eigen>
#include <cstdint>
#include <iostream>
#include <thrust/detail/config/host_device.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#ifdef PBAL_NEIGHBOR_SEARCH_CHECK
#define PBAL_NEIGHBOR_SEARCH_ASSERT(x) PBAL_NEIGHBOR_SEARCH_ASSERT(x)
#else
#define PBAL_NEIGHBOR_SEARCH_ASSERT(x) ((void)0)
#endif

// --- declaration start ---

namespace pbal {
struct Neighbors {
  thrust::device_vector<uint32_t> offset;
  thrust::device_vector<uint32_t> counter;
  thrust::device_vector<uint32_t> neighbors;
};

template <typename real, std::size_t dim> struct ParticleDistributionInCell;

template <typename real, std::size_t dim>
auto findNeighbors(Eigen::Vector<real, dim> const *pointA, uint32_t const a,
                   real const spacing,
                   Eigen::Vector<real, dim> const *queryPointD,
                   uint32_t const d) -> Neighbors;

template <typename real, std::size_t dim>
auto computeCellInformation(
    thrust::device_ptr<Eigen::Vector<real, dim> const> const pointA,
    uint32_t const a, real const spacing,
    ParticleDistributionInCell<real, dim> &output) -> void;

template <typename real, std::size_t dim>
auto findNeighborsForParticleDistribution(
    Eigen::Vector<real, dim> const *pointA, uint32_t const a,
    ParticleDistributionInCell<real, dim> const &cellInformation,
    Eigen::Vector<real, dim> const *queryPointD, uint32_t const d,
    Neighbors &neighbors) -> void;

template <typename real, std::size_t dim>
auto findNeighbors(Eigen::Vector<real, dim> const *pointA, uint32_t const a,
                   real const spacing,
                   Eigen::Vector<real, dim> const *queryPointD,
                   uint32_t const d,
                   ParticleDistributionInCell<real, dim> &cellInformaitonBuffer,
                   Neighbors &output) -> void;
} // namespace pbal

// --- declaration end ---

// --- implementation start ---

namespace pbal {
template <typename real, std::size_t dim> struct BoundBox {
  Eigen::Vector<real, dim> min, max;
};

template <typename real, std::size_t dim> struct CellParam {
  Eigen::Vector<uint32_t, dim> gridDimension;
  uint32_t particleCount;
  BoundBox<real, dim> boundBox;
  real spacing;
};

template <typename real, std::size_t dim> struct ParticleDistributionInCell {
  CellParam<real, dim> param;
  thrust::device_vector<uint32_t> particleCellIdxA, particleOrderA,
      mapFromOrderToParticleA;
  thrust::device_vector<uint32_t> countB, offsetB;
};
template <typename real, std::size_t dim>
auto computeBoundBox(thrust::device_ptr<Eigen::Vector<real, dim> const> pointA,
                     uint32_t n) -> BoundBox<real, dim> {
  using Vec = Eigen::Vector<real, dim>;
  using Box = BoundBox<real, dim>;
  PBAL_NEIGHBOR_SEARCH_ASSERT(n > 0);
  auto const init = Box{*pointA, *pointA};
  return thrust::transform_reduce(
      pointA, pointA + n,
      [] __device__ __host__(Vec v) -> Box {
        return Box{v, v};
      },
      init,
      [] __device__ __host__(Box l, Box r) -> Box {
        Vec const lower = l.min.array().min(r.min.array()),
                  upper = l.max.array().max(r.max.array());
        return Box{lower, upper};
      });
}

template <typename real, std::size_t dim>
auto ceilBoundBox(BoundBox<real, dim> boundBox, real spacing)
    -> std::tuple<BoundBox<real, dim>, Eigen::Vector<uint32_t, dim>> {
  // ceil boundBox according to spacing
  auto const gridDimension = [boundBox, spacing] {
    auto res = Eigen::Vector<uint32_t, dim>();
    auto const [mi, ma] = boundBox;
    for (auto i = decltype(dim)(0); i < dim; ++i) {
      res[i] = uint32_t(std::ceil((ma[i] - mi[i]) / spacing));
    }
    return res;
  }();

  // ceil boundBox.max
  auto const gridMax = [boundBox, gridDimension, spacing] {
    auto res = Eigen::Vector<real, dim>();
    auto const [mi, ma] = boundBox;
    for (auto i = decltype(dim)(0); i < dim; ++i) {
      res[i] = mi[i] + gridDimension[i] * spacing;
    }
    return res;
  }();

  return std::tuple{BoundBox<real, dim>{boundBox.min, gridMax}, gridDimension};
}

template <typename real, std::size_t dim>
auto computeCellParam(
    thrust::device_ptr<Eigen::Vector<real, dim> const> const pointA,
    uint32_t const n, real const spacing) -> CellParam<real, dim> {
  auto const particleBoundBox = computeBoundBox<real, dim>(pointA, n);
  auto const [boundBox, gridDimension] =
      ceilBoundBox<real, dim>(particleBoundBox, spacing);
  auto const particleCount = n;
  return CellParam<real, dim>{gridDimension, particleCount, boundBox, spacing};
}

template <typename real, std::size_t dim>
auto padCellParam(CellParam<real, dim> cellParam) -> CellParam<real, dim> {
  for (auto i = decltype(dim)(0); i < dim; ++i) {
    cellParam.boundBox.min[i] -= 2 * cellParam.spacing;
    cellParam.boundBox.max[i] += 2 * cellParam.spacing;
    cellParam.gridDimension[i] += 4;
  }
  return cellParam;
}

template <typename T, std::size_t dim>
auto product(Eigen::Vector<T, dim> v) -> T {
  auto res = static_cast<T>(1);
  for (auto i = decltype(dim)(0); i < dim; ++i) {
    res *= v[i];
  }
  return res;
}

template <std::size_t dim>
__device__ __host__ auto cellToIndex(Eigen::Vector<uint32_t, dim> gridDimension,
                                     Eigen::Vector<uint32_t, dim> cell)
    -> uint32_t {
  auto res = uint32_t(0);
  for (auto i = decltype(dim)(0); i < dim; ++i) {
    res *= gridDimension[dim - 1 - i];
    res += cell[dim - 1 - i];
  }
  return res;
}

template <typename real, std::size_t dim>
__host__ __device__ auto computeCell(Eigen::Vector<real, dim> gridMin,
                                     real spacing, Eigen::Vector<real, dim> pos)
    -> Eigen::Vector<uint32_t, dim> {
  auto const relativePosition = decltype(pos)(pos - gridMin);
  auto const cellF =
      decltype(pos)((relativePosition / spacing).array().floor());
  auto const cell = [cellF] __device__ __host__ {
    auto res = Eigen::Vector<uint32_t, dim>();
    for (auto i = decltype(dim)(0); i < dim; ++i) {
      res[i] = uint32_t(cellF[i]);
    }
    return res;
  }();
  return cell;
}

template <std::size_t dim, std::size_t recurDepth, typename Op>
__host__ __device__ auto
_tranverseImplement(Eigen::Vector<uint32_t, dim> const cell, Op const &op)
    -> void {
  if constexpr (recurDepth < dim) {
    auto const recurDim = dim - 1 - recurDepth;
    for (auto i = int32_t(-1); i < 2; ++i) {
      auto const newCell = [cell, i] {
        auto res = cell;
        PBAL_NEIGHBOR_SEARCH_ASSERT(res[recurDim] > 0);
        res[recurDim] += i;
        return res;
      }();
      _tranverseImplement<dim, recurDepth + 1, Op>(newCell, op);
    }
  } else {
    op(cell);
  }
}

// this function doesn't handle boundary!!! so padding around bound box is
// necessary
template <std::size_t dim, typename Op>
__host__ __device__ auto
traverseAroundCellNeighbor(Eigen::Vector<uint32_t, dim> center, Op op) -> void {
  _tranverseImplement<dim, 0, Op>(center, op);
}

template <typename real, std::size_t dim>
auto computeCellInformation(
    thrust::device_ptr<Eigen::Vector<real, dim> const> const pointA,
    uint32_t const a, real const spacing,
    ParticleDistributionInCell<real, dim> &cellBuffer) -> void {
  using Vec = Eigen::Vector<real, dim>;
  using Cell = Eigen::Vector<uint32_t, dim>;
  // compute cell param
  auto const unpaddedCellParam =
      computeCellParam<real, dim>(pointA, a, spacing);
  // pad cell param to avoid boundary check
  auto const cellParam = padCellParam<real, dim>(unpaddedCellParam);
  cellBuffer.param = cellParam;
  auto const numberOfCells =
      product<uint32_t, dim>(cellBuffer.param.gridDimension);

  // allocate cell memory
  cellBuffer.particleCellIdxA.resize(a);
  cellBuffer.particleOrderA.resize(a);
  cellBuffer.mapFromOrderToParticleA.resize(a);
  cellBuffer.countB.resize(numberOfCells);
  cellBuffer.offsetB.resize(numberOfCells);

  thrust::fill(cellBuffer.particleOrderA.begin(),
               cellBuffer.particleOrderA.end(), uint32_t(0));
  thrust::fill(cellBuffer.countB.begin(), cellBuffer.countB.end(), 0);

  // compute idx of cell of particle
  // compute cell count
  // particleOrderA is fuse for two purpose: particleOrderInCell and
  // particleOrder std::cout << "01 compute cell count" << std::endl;
  thrust::for_each(
      thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(a),
      [particleCellIdxA =
           thrust::raw_pointer_cast(cellBuffer.particleCellIdxA.data()),
       cellCountB = thrust::raw_pointer_cast(cellBuffer.countB.data()),
       pointA = pointA,
       particleOrderA =
           thrust::raw_pointer_cast(cellBuffer.particleOrderA.data()),
       gridDimension = cellParam.gridDimension,
       gridMin = cellParam.boundBox.min,
       spacing = cellParam.spacing] __device__(uint32_t particleIdx) -> void {
        auto const cell =
            computeCell<real, dim>(gridMin, spacing, pointA[particleIdx]);
        for (auto i = decltype(dim)(0); i < dim; ++i) {
          PBAL_NEIGHBOR_SEARCH_ASSERT(cell[i] < gridDimension[i]);
        }
        auto const index = cellToIndex<dim>(gridDimension, cell);
        particleCellIdxA[particleIdx] = index;
        particleOrderA[particleIdx] = atomicAdd(cellCountB + index, 1);
      });

  // compute offset
  // std::cout << "02 compute cell offset" << std::endl;
  thrust::exclusive_scan(cellBuffer.countB.begin(), cellBuffer.countB.end(),
                         cellBuffer.offsetB.begin());

  PBAL_NEIGHBOR_SEARCH_ASSERT(
      cellBuffer.countB.back() + cellBuffer.offsetB.back() == a);

  // std::cout << "03 compute cellBuffer.particleOrderA" << std::endl;
  thrust::transform(
      cellBuffer.particleCellIdxA.begin(), cellBuffer.particleCellIdxA.end(),
      cellBuffer.particleOrderA.begin(), cellBuffer.particleOrderA.begin(),
      [cellOffsetB = thrust::raw_pointer_cast(
           cellBuffer.offsetB
               .data())] __device__(uint32_t particleCellIdx,
                                    uint32_t orderOfParticleInCell)
          -> uint32_t {
        return cellOffsetB[particleCellIdx] + orderOfParticleInCell;
      });

  // std::cout << "04 compute mapFromOrderToParticleC" << std::endl;
  thrust::scatter(thrust::counting_iterator<uint32_t>(0),
                  thrust::counting_iterator<uint32_t>(a),
                  cellBuffer.particleOrderA.begin(),
                  cellBuffer.mapFromOrderToParticleA.begin());
}

template <typename real, std::size_t dim>
auto findNeighborsForParticleDistribution(
    Eigen::Vector<real, dim> const *pointA, uint32_t const a,
    ParticleDistributionInCell<real, dim> const &cellInformation,
    Eigen::Vector<real, dim> const *queryPointD, uint32_t const d,
    Neighbors &neighbors) -> void {
  using Vec = Eigen::Vector<real, dim>;
  using Cell = Eigen::Vector<uint32_t, dim>;

  auto const cellParam = cellInformation.param;
  auto const numberOfCells = cellInformation.countB.size();

  neighbors.counter.resize(d);
  thrust::fill(neighbors.counter.begin(), neighbors.counter.end(), 0);
  neighbors.offset.resize(d);

  auto wrappedQueryPointD = thrust::device_ptr<Vec const>(queryPointD);

  // gpu 跟 cpu
  // 有不同的舍入算法，所以算出来所处的格子可能不一样，但这个影响不大。

  // std::cout << "05 counting neighbors" << std::endl;
  thrust::transform(
      wrappedQueryPointD, wrappedQueryPointD + d, neighbors.counter.begin(),
      [=, cellCountB = thrust::raw_pointer_cast(cellInformation.countB.data()),
       cellOffsetB = thrust::raw_pointer_cast(cellInformation.offsetB.data()),
       mapFromOrderToParticleA = thrust::raw_pointer_cast(
           cellInformation.mapFromOrderToParticleA.data()),
       pointA = pointA, gridDimension = cellParam.gridDimension,
       gridMin = cellParam.boundBox.min,
       spacing = cellParam.spacing] __device__(Vec p) -> uint32_t {
        auto const center = computeCell<real, dim>(gridMin, spacing, p);
        auto neighborCount = uint32_t(0);
#ifdef PBAL_NEIGHBOR_SEARCH_CHECK
        for (auto i = decltype(dim)(0); i < dim; ++i) {
          PBAL_NEIGHBOR_SEARCH_ASSERT(center[i] > 0);
          PBAL_NEIGHBOR_SEARCH_ASSERT(center[i] + 1 < gridDimension[i]);
        }
#endif
        traverseAroundCellNeighbor<dim>(
            center, [=, &neighborCount] __device__(Cell cell) -> void {
              auto const cellIdx = cellToIndex<dim>(gridDimension, cell);
              PBAL_NEIGHBOR_SEARCH_ASSERT(cellIdx < numberOfCells);
              auto const cellCount = cellCountB[cellIdx];
              auto const cellStart = cellOffsetB[cellIdx];
              for (auto i = cellStart; i < cellStart + cellCount; ++i) {
                PBAL_NEIGHBOR_SEARCH_ASSERT(i < a);
                auto const particleIdx = mapFromOrderToParticleA[i];
                PBAL_NEIGHBOR_SEARCH_ASSERT(particleIdx < a);
                auto const point = pointA[particleIdx];
                auto const diff = decltype(p)(point - p);
                auto const distance2 = diff.dot(diff);
                if (distance2 < spacing * spacing) {
                  ++neighborCount;
                }
              }
            });
        return neighborCount;
      });

  // compute neighbors offset
  // std::cout << "06" << std::endl;
  thrust::exclusive_scan(neighbors.counter.begin(), neighbors.counter.end(),
                         neighbors.offset.begin());

  auto const totalNeighborCount =
      neighbors.counter.back() + neighbors.offset.back();

  // std::cout << "07" << std::endl;
  neighbors.neighbors.resize(totalNeighborCount);

  // fill neighbors
  // std::cout << "08" << std::endl;
  thrust::for_each_n(
      thrust::counting_iterator<uint32_t>(0), d,
      [countOfCellB = thrust::raw_pointer_cast(cellInformation.countB.data()),
       offsetOfCellB = thrust::raw_pointer_cast(cellInformation.offsetB.data()),
       mapFromOrderToParticleA = thrust::raw_pointer_cast(
           cellInformation.mapFromOrderToParticleA.data()),
       pointA = pointA, queryPointD = queryPointD,
       queryPointNeighborOffsetD =
           thrust::raw_pointer_cast(neighbors.offset.data()),
       neighborsOfQueryPoint =
           thrust::raw_pointer_cast(neighbors.neighbors.data()),
       gridDimension = cellParam.gridDimension,
       gridMin = cellParam.boundBox.min,
       spacing = cellParam.spacing] __device__(uint32_t queryPointIdx) -> void {
        auto const p = queryPointD[queryPointIdx];
        auto const center = computeCell<real, dim>(gridMin, spacing, p);
        auto fillOffset = queryPointNeighborOffsetD[queryPointIdx];
        traverseAroundCellNeighbor<dim>(
            center, [=, &fillOffset] __device__(Cell cell) {
              auto const cellIdx = cellToIndex<dim>(gridDimension, cell);
              auto const cellCount = countOfCellB[cellIdx];
              auto const cellStart = offsetOfCellB[cellIdx];
              for (auto i = cellStart; i < cellStart + cellCount; ++i) {
                auto const particleIdx = mapFromOrderToParticleA[i];
                auto const point = pointA[particleIdx];
                auto const diff = point - p;
                auto const distance2 = diff.dot(diff);
                if (distance2 < spacing * spacing) {
                  neighborsOfQueryPoint[fillOffset] = particleIdx;
                  ++fillOffset;
                }
              }
            });
      });
}

template <typename real, std::size_t dim>
auto findNeighbors(Eigen::Vector<real, dim> const *pointA, uint32_t const a,
                   real const spacing,
                   Eigen::Vector<real, dim> const *queryPointD,
                   uint32_t const d,
                   ParticleDistributionInCell<real, dim> &cellInformaitonBuffer,
                   Neighbors &output) -> void {
  auto const wrapperedPointA =
      thrust::device_ptr<Eigen::Vector<real, dim> const>(pointA);
  computeCellInformation<real, dim>(wrapperedPointA, a, spacing,
                                    cellInformaitonBuffer);
  findNeighborsForParticleDistribution<real, dim>(
      pointA, a, cellInformaitonBuffer, queryPointD, d, output);
}

template <typename real, std::size_t dim>
auto findNeighbors(Eigen::Vector<real, dim> const *pointA, uint32_t const a,
                   real const spacing,
                   Eigen::Vector<real, dim> const *queryPointD,
                   uint32_t const d) -> Neighbors {

  auto cellBuffer = ParticleDistributionInCell<real, dim>();
  auto neighbors = Neighbors{};

  findNeighbors<real, dim>(pointA, a, spacing, queryPointD, d, cellBuffer,
                           neighbors);

  return neighbors;
}
} // namespace pbal

// --- implementation end ---
