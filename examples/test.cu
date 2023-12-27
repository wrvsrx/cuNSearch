#include "compute_neighbor.cuh"
#include "cuNSearch.h"
#include <Eigen/Eigen>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <vector>

// using real = double;
// using Vec = Eigen::Vector<real, 3>;
// auto const dim = static_cast<std::size_t>(3);

template <std::size_t dim, std::size_t recurDepth, typename Op>
__host__ __device__ auto
_tranverseImplement(std::array<uint32_t, dim> const posAndRange, Op const &op)
    -> void {
  if constexpr (recurDepth < dim) {
    auto const recurDim = dim - 1 - recurDepth;
    for (auto i = uint32_t(0); i < posAndRange[recurDim]; ++i) {
      auto const newPosAndRange = [posAndRange, i] {
        auto res = posAndRange;
        res[recurDim] = i;
        return res;
      }();
      _tranverseImplement<dim, recurDepth + 1, Op>(newPosAndRange, op);
    }
  } else {
    op(posAndRange);
  }
}

template <std::size_t dim, typename Op>
__host__ __device__ auto
traverseAroundCellNeighbor(std::array<uint32_t, dim> range, Op op) -> void {
  _tranverseImplement<dim, 0, Op>(range, op);
}

template <typename real, std::size_t dim>
auto initGrid(uint32_t const n, real r)
    -> std::vector<Eigen::Vector<real, dim>> {
  using Vec = Eigen::Vector<real, dim>;
  auto res = std::vector<Vec>();
  auto const num = [=] {
    auto res = uint32_t(1);
    for (auto i = decltype(dim)(0); i < dim; ++i) {
      res *= n;
    }
    return res;
  }();
  res.reserve(num);
  auto const range = [=] {
    auto res = std::array<uint32_t, dim>();
    std::fill(res.begin(), res.end(), n);
    return res;
  }();

  traverseAroundCellNeighbor(range,
                             [=, &res](std::array<uint32_t, dim> pos) -> void {
                               auto v = Vec();
                               for (auto i = decltype(dim)(0); i < dim; ++i) {
                                 v[i] = r * pos[i];
                               }
                               res.push_back(v);
                             });

  return res;
}

auto main() -> int {
  using real = double;
  auto constexpr dim = std::size_t(3);
  auto constexpr r = static_cast<double>(0.15);
  auto const grid = initGrid<real, dim>(100, r);
  auto const dGrid = thrust::device_vector<Eigen::Vector<real, dim>>(grid);
  auto const pointA = thrust::raw_pointer_cast(dGrid.data());
  {
    auto const startTime = std::chrono::high_resolution_clock::now();
    auto const dGrid = thrust::device_vector<Eigen::Vector<real, dim>>(grid);
    pbal::findNeighbors<real, dim>(pointA, grid.size(), 2 * r, pointA,
                                   grid.size());
    auto const endTime = std::chrono::high_resolution_clock::now();
    auto const durationTime =
        std::chrono::duration<double>(endTime - startTime);
    std::cout << "time cost: " << durationTime.count() << std::endl;
  }

  {
    auto const startTime = std::chrono::high_resolution_clock::now();
    cuNSearch::NeighborhoodSearch nsearch(2 * r);
    nsearch.add_point_set((real const *)(grid.data()), grid.size());
    nsearch.find_neighbors();
    auto const endTime = std::chrono::high_resolution_clock::now();
    auto const durationTime =
        std::chrono::duration<double>(endTime - startTime);
    std::cout << "time cost: " << durationTime.count() << std::endl;
  }
}
