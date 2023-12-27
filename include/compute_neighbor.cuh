// input: points, radius
// output: hashed grid with point information

// input: hashed grid with point information, position
// output: count around position

// input: hashed grid with point information, positions and their count
// output: neighbored list

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
#include <tuple>
#include <type_traits>

namespace pbal {
template <std::size_t dim> using Cell = Eigen::Vector<uint32_t, dim>;

struct Neighbors {
  thrust::device_vector<uint32_t> offset;
  thrust::device_vector<uint32_t> neighbors;
};

} // namespace pbal

namespace pbal {
template <typename real, std::size_t dim> struct BoundBox {
  Eigen::Vector<real, dim> min, max;
};

template <typename real, std::size_t dim>
auto computeBoundBox(thrust::device_ptr<Eigen::Vector<real, dim> const> pointA,
                     uint32_t n) -> BoundBox<real, dim> {
  using Vec = Eigen::Vector<real, dim>;
  using Box = BoundBox<real, dim>;
  assert(n > 0);
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

template <typename real, std::size_t dim> struct CellParam {
  Eigen::Vector<uint32_t, dim> gridDimension;
  uint32_t particleCount;
  BoundBox<real, dim> boundBox;
  real spacing;
};

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
    cellParam.boundBox.min[i] -= cellParam.spacing;
    cellParam.boundBox.max[i] += cellParam.spacing;
    cellParam.gridDimension[i] += 2;
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
  auto res = static_cast<uint32_t>(0);
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
        assert(res[recurDim] > 0);
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
auto findNeighbors(Eigen::Vector<real, dim> const *pointA, uint32_t const a,
                   real const spacing,
                   Eigen::Vector<real, dim> const *queryPointD,
                   uint32_t const d) -> Neighbors {
  using Vec = Eigen::Vector<real, dim>;
  using Cell = Eigen::Vector<uint32_t, dim>;
  auto const wrapperedPointA = thrust::device_ptr<Vec const>(pointA);
  // compute cell param
  auto const unpaddedCellParam =
      computeCellParam<real, dim>(wrapperedPointA, a, spacing);
  // pad cell param to avoid boundary check
  auto const cellParam = padCellParam<real, dim>(unpaddedCellParam);
  auto const numberOfCells = product<uint32_t, dim>(cellParam.gridDimension);

  // allocate cell memory
  auto cellOffsetB = thrust::device_vector<uint32_t>(numberOfCells);
  auto cellCountB = thrust::device_vector<uint32_t>(numberOfCells, 0);

  auto particleCellIdxA = thrust::device_vector<uint32_t>(a);
  auto sortIdxA = thrust::device_vector<uint32_t>(a);
  auto reversedSortIdxA = thrust::device_vector<uint32_t>(a);

  auto particleOrderInCellA = thrust::device_vector<uint32_t>(a);

  // compute idx of cell of particle
  // compute cell count
  std::cout << "01 compute cell count" << std::endl;
  thrust::for_each(
      thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(a),
      [particleCellIdxA = thrust::raw_pointer_cast(particleCellIdxA.data()),
       cellCountB = thrust::raw_pointer_cast(cellCountB.data()),
       pointA = pointA,
       particleOrderInCellA =
           thrust::raw_pointer_cast(particleOrderInCellA.data()),
       gridDimension = cellParam.gridDimension,
       gridMin = cellParam.boundBox.min,
       spacing = cellParam.spacing] __device__(uint32_t particleIdx) -> void {
        auto const cell =
            computeCell<real, dim>(gridMin, spacing, pointA[particleIdx]);
        for (auto i = decltype(dim)(0); i < dim; ++i) {
          assert(cell[i] < gridDimension[i]);
        }
        auto const index = cellToIndex<dim>(gridDimension, cell);
        particleCellIdxA[particleIdx] = index;
        particleOrderInCellA[particleIdx] = atomicAdd(cellCountB + index, 1);
      });

  // compute offset
  std::cout << "02 compute cell offset" << std::endl;
  thrust::exclusive_scan(cellCountB.begin(), cellCountB.end(),
                         cellOffsetB.begin());

  assert(cellCountB.back() + cellOffsetB.back() == a);

  // this can fuse orderOfParticleInCellA, for code clear we doesn't do that
  auto orderOfParticleAccordingToCellA = thrust::device_vector<uint32_t>(a);
  auto mapFromOrderToParticleC = thrust::device_vector<uint32_t>(a);

  std::cout << "03 compute orderOfParticleAccordingToCellA" << std::endl;
  thrust::transform(
      particleCellIdxA.begin(), particleCellIdxA.end(),
      particleOrderInCellA.begin(), orderOfParticleAccordingToCellA.begin(),
      [offsetOfCellB = thrust::raw_pointer_cast(cellOffsetB.data())] __device__(
          uint32_t idxOfCellOfParticle,
          uint32_t orderOfParticleInCell) -> uint32_t {
        return offsetOfCellB[idxOfCellOfParticle] + orderOfParticleInCell;
      });

  std::cout << "03.5 " << std::endl;
  thrust::for_each(cellOffsetB.begin(), cellOffsetB.end(),
                   [=] __device__(uint32_t idx) { assert(idx <= a); });
  std::cout << "03.55 " << std::endl;
  thrust::for_each_n(thrust::counting_iterator<uint32_t>(0), a,
                     [orderOfParticleInCellA =
                          thrust::raw_pointer_cast(particleOrderInCellA.data()),
                      cellCountB = thrust::raw_pointer_cast(cellCountB.data()),
                      idxOfCellOfParticleA = thrust::raw_pointer_cast(
                          particleCellIdxA.data())] __device__(uint32_t idx) {
                       auto const cellIdx = idxOfCellOfParticleA[idx];
                       auto const cellCount = cellCountB[cellIdx];
                       assert(orderOfParticleInCellA[idx] < cellCount);
                     });
  thrust::for_each(particleOrderInCellA.begin(), particleOrderInCellA.end(),
                   [=] __device__(uint32_t idx) { assert(idx < a); });
  std::cout << "03.6 " << std::endl;
  thrust::for_each(orderOfParticleAccordingToCellA.begin(),
                   orderOfParticleAccordingToCellA.end(),
                   [=] __device__(uint32_t idx) { assert(idx < a); });

  std::cout << "04 compute mapFromOrderToParticleC" << std::endl;
  thrust::gather(orderOfParticleAccordingToCellA.begin(),
                 orderOfParticleAccordingToCellA.end(),
                 thrust::counting_iterator<uint32_t>(0),
                 mapFromOrderToParticleC.begin());

  std::cout << "05 counting neighbors" << std::endl;
  auto particleNeighborCountD = thrust::device_vector<uint32_t>(d, 0);
  auto wrappedQueryPointD = thrust::device_ptr<Vec const>(queryPointD);
  auto queryPointNeighborCountD = thrust::device_vector<uint32_t>(d);
  thrust::transform(
      wrappedQueryPointD, wrappedQueryPointD + d,
      queryPointNeighborCountD.begin(),
      [=, cellCountB = thrust::raw_pointer_cast(cellCountB.data()),
       cellOffsetB = thrust::raw_pointer_cast(cellOffsetB.data()),
       mapFromOrderToParticleC =
           thrust::raw_pointer_cast(mapFromOrderToParticleC.data()),
       pointA = pointA, gridDimension = cellParam.gridDimension,
       gridMin = cellParam.boundBox.min,
       spacing = cellParam.spacing] __device__(Vec p) -> uint32_t {
        auto const center = computeCell<real, dim>(gridMin, spacing, p);
        auto neighborCount = uint32_t(0);
        assert(center[0] > 0);
        assert(center[1] > 0);
        assert(center[2] > 0);
        traverseAroundCellNeighbor<dim>(
            center, [=, &neighborCount] __device__(Cell cell) -> void {
              auto const cellIdx = cellToIndex<dim>(gridDimension, cell);
              assert(cellIdx < numberOfCells);
              auto const cellCount = cellCountB[cellIdx];
              auto const cellStart = cellOffsetB[cellIdx];
              for (auto i = cellStart; i < cellStart + cellCount; ++i) {
                assert(i < a);
                auto const particleIdx = mapFromOrderToParticleC[i];
                assert(particleIdx < a);
                auto const point = pointA[particleIdx];
                auto const diff = decltype(p)(point - p);
                auto const distance2 = diff.dot(diff);
                if (distance2 < spacing * spacing &&
                    distance2 > static_cast<real>(0.0)) {
                  ++neighborCount;
                }
              }
            });
        return neighborCount;
      });

  // compute neighbors offset
  std::cout << "06" << std::endl;
  auto queryPointNeighborOffsetD = thrust::device_vector<uint32_t>(d);
  thrust::exclusive_scan(queryPointNeighborCountD.begin(),
                         queryPointNeighborCountD.end(),
                         queryPointNeighborOffsetD.begin());

  auto const totalNeighborCount =
      queryPointNeighborOffsetD.back() + queryPointNeighborCountD.back();

  std::cout << "07" << std::endl;
  auto neighborsOfQueryPoint =
      thrust::device_vector<uint32_t>(totalNeighborCount);

  // fill neighbors
  std::cout << "08" << std::endl;
  thrust::for_each_n(
      thrust::counting_iterator<uint32_t>(0), d,
      [countOfCellB = thrust::raw_pointer_cast(cellCountB.data()),
       offsetOfCellB = thrust::raw_pointer_cast(cellOffsetB.data()),
       mapFromOrderToParticleC =
           thrust::raw_pointer_cast(mapFromOrderToParticleC.data()),
       pointA = pointA, queryPointD = queryPointD,
       queryPointNeighborOffsetD =
           thrust::raw_pointer_cast(queryPointNeighborOffsetD.data()),
       neighborsOfQueryPoint =
           thrust::raw_pointer_cast(neighborsOfQueryPoint.data()),
       gridDimension = cellParam.gridDimension,
       gridMin = cellParam.boundBox.min,
       spacing = cellParam.spacing] __device__(uint32_t queryPointIdx) -> void {
        auto const p = queryPointD[queryPointIdx];
        auto const cell = computeCell<real, dim>(gridMin, spacing, p);
        auto fillOffset = queryPointNeighborOffsetD[queryPointIdx];
        traverseAroundCellNeighbor<dim>(
            cell, [=, &fillOffset] __device__(Cell cell) {
              auto const cellIdx = cellToIndex<dim>(gridDimension, cell);
              auto const cellCount = countOfCellB[cellIdx];
              auto const cellStart = offsetOfCellB[cellIdx];
              for (auto i = cellStart; i < cellStart + cellCount; ++i) {
                auto const particleIdx = mapFromOrderToParticleC[i];
                auto const point = pointA[particleIdx];
                auto const diff = point - p;
                auto const distance2 = diff.dot(diff);
                if (distance2 < spacing * spacing &&
                    distance2 > static_cast<real>(0.0)) {
                  neighborsOfQueryPoint[fillOffset] = particleIdx;
                  ++fillOffset;
                }
              }
            });
      });
  return Neighbors{queryPointNeighborOffsetD, neighborsOfQueryPoint};
}
} // namespace pbal
