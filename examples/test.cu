#include "compare-two-neighbors.hpp"
#include "compute_neighbor.cuh"
#include "cuNSearch.h"
#include "for-in-multiple-range.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cstdint>
#include <fmt/format.h>
#include <iterator>
#include <random>
#include <string>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <vector>
template <typename T, std::size_t N> auto constexpr pow(T val) -> T {
  if constexpr (N == 0)
    return static_cast<T>(1);
  else if constexpr (N == 1)
    return val;
  auto half = pow<T, N / 2>(val);
  if constexpr (N % 2 == 0)
    return half * half;
  else
    return val * half * half;
}

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
    -> thrust::device_vector<Eigen::Vector<real, dim>> {
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
  auto constexpr spacing = 1.2 * r;
  auto constexpr n = uint32_t(100);
  auto constexpr totalNum = pow<uint32_t, dim>(n);
  Eigen::IOFormat const CommaInitFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
  auto engine = std::mt19937_64(114514);
  std::cout << std::uniform_real_distribution<real>(0.0, 1.0)(engine)
            << std::endl;
  auto normalDis = std::uniform_real_distribution<real>(0.0, 1.0);
  auto const gen = [&]() -> double { return normalDis(engine); };
  auto const randomSequence = [&] {
    std::vector<real> res(totalNum * dim);
    std::generate(res.begin(), res.end(), gen);
    return res;
  }();

  std::cout << "finish generating sequence" << std::endl;

  auto const dPointA = [&] {
    auto const dRandomSequence = thrust::device_vector<real>(randomSequence);
    auto const randomSequencePtr =
        thrust::device_ptr<Eigen::Vector<real, dim> const>(
            (Eigen::Vector<real, dim> const *)(thrust::raw_pointer_cast(
                dRandomSequence.data())));
    auto res = thrust::device_vector<Eigen::Vector<real, dim>>(totalNum);
    thrust::transform(
        randomSequencePtr, randomSequencePtr + totalNum, res.begin(),
        [=] __device__(Eigen::Vector<real, dim> d) { return r * d * n; });
    return res;
  }();

  // auto const dPointA = [] {
  //   auto res = initGrid<real, dim>(n, r);
  //   thrust::default_random_engine g(1);
  //   thrust::shuffle(res.begin() + 1, res.end(), g);
  //   return res;
  // }();

  std::cout << "finish generating points" << std::endl;

  auto const hPointA = thrust::host_vector<Eigen::Vector<real, dim>>(dPointA);

  auto cellInformation = pbal::ParticleDistributionInCell<real, dim>();
  auto neighbors = pbal::Neighbors();
  {
    auto const startTime = std::chrono::high_resolution_clock::now();
    auto const pPointA = thrust::raw_pointer_cast(dPointA.data());
    pbal::findNeighbors<real, dim>(pPointA, totalNum, spacing, pPointA,
                                   totalNum, cellInformation, neighbors);
    auto const endTime = std::chrono::high_resolution_clock::now();
    auto const durationTime =
        std::chrono::duration<double>(endTime - startTime);
    std::cout << "time cost: " << durationTime.count() << std::endl;
  }
  auto const firstNeighborHost = NeighborsHost{
      neighbors.offset,
      neighbors.counter,
      neighbors.neighbors,
  };

  auto const dump = [&](std::string const &str, NeighborsHost const &neighbor) {
    std::cout << "----- dump start -----\n"
              << str << '\n'
              << neighbor.offsetA.size() << '\n'
              << neighbor.counterA.size() << '\n'
              << neighbor.neighborB.size() << '\n'
              << neighbor.offsetA[0] << '\n'
              << neighbor.counterA[0] << '\n';
    std::cout << "first point: " << hPointA[0].format(CommaInitFmt) << '\n'
              << fmt::format("first point neighbor num: {}\n",
                             neighbor.counterA[0])
              << "first point's neighbors:\n";
    for (auto i = neighbor.offsetA[0];
         i < neighbor.offsetA[0] + neighbor.counterA[0]; ++i) {
      std::cout << "neighbor " << i << ": "
                << hPointA[neighbor.neighborB[i]].format(CommaInitFmt) << '\n';
    }
    std::cout << "----- dump end -----\n";
  };
  dump("first", firstNeighborHost);

  cuNSearch::NeighborhoodSearch nsearch(spacing);
  nsearch.add_point_set((real const *)(hPointA.data()), totalNum);
  {
    auto const startTime = std::chrono::high_resolution_clock::now();
    nsearch.find_neighbors();
    auto const endTime = std::chrono::high_resolution_clock::now();
    auto const durationTime =
        std::chrono::duration<double>(endTime - startTime);
    std::cout << "time cost: " << durationTime.count() << std::endl;
  }

  auto const secondNeighborHost = [&nsearch] {
    auto res = NeighborsHost{};
    auto const &ps = nsearch.point_set(0);
    auto const num = ps.n_points();
    res.counterA.resize(num);
    res.offsetA.resize(num);
    for (auto i = 0; i < ps.n_points(); ++i) {
      auto const neighborNum = ps.n_neighbors(0, i);
      res.counterA[i] = neighborNum;
      for (auto j = 0; j < neighborNum; ++j) {
        auto const pid = ps.neighbor(0, i, j);
        res.neighborB.push_back(pid);
      }
    }
    thrust::exclusive_scan(res.counterA.begin(), res.counterA.end(),
                           res.offsetA.begin());
    return res;
  }();
  dump("second", secondNeighborHost);

  checkNeighborConsistent<real, dim>(firstNeighborHost, secondNeighborHost,
                                     spacing, hPointA, hPointA);
}
