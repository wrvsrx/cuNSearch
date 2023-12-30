#pragma once
#include "compute_neighbor.cuh"
#include <Eigen/src/Core/IO.h>
#include <iomanip>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <type_traits>

struct NeighborsHost {
  thrust::host_vector<uint32_t> offsetA;
  thrust::host_vector<uint32_t> counterA;
  thrust::host_vector<uint32_t> neighborB;
};

auto inline checkNeighbor(NeighborsHost const &neighbors) -> void {
  assert(neighbors.offsetA.size() == neighbors.counterA.size());
}

template <typename real, std::size_t dim>
auto inline checkNeighborConsistent(
    NeighborsHost const &l, NeighborsHost const &r, real const spacing,
    thrust::host_vector<Eigen::Vector<real, dim>> const &pointA,
    thrust::host_vector<Eigen::Vector<real, dim>> const &queryPointB) -> void {
  checkNeighbor(l);
  checkNeighbor(r);

  std::cout << l.offsetA.size() << ' ' << r.offsetA.size() << std::endl;
  assert(l.offsetA.size() == queryPointB.size());
  assert(r.offsetA.size() == queryPointB.size());
  std::cout << l.neighborB.size() << ' ' << r.neighborB.size() << std::endl;

  auto const a = l.offsetA.size();

  auto lNeighborB = l.neighborB, rNeighborB = r.neighborB;

  Eigen::IOFormat const CommaInitFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");

  thrust::for_each(
      thrust::host, thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(a), [&](uint32_t idx) {
        auto const q = queryPointB[idx];
        auto const computeDiff = [spacing](Eigen::Vector<real, dim> p,
                                           Eigen::Vector<real, dim> q) {
          auto const spacing2 = spacing * spacing;
          auto const diff = (spacing2 - (p - q).dot(p - q)) / spacing2;
          return diff;
        };
        auto const filterNeighbor = [&](NeighborsHost const &neighbor) {
          auto const start = neighbor.offsetA[idx],
                     end = start + neighbor.counterA[idx];
          auto res = thrust::host_vector<uint32_t>(end - start);
          auto const ite = thrust::copy_if(neighbor.neighborB.begin() + start,
                                           neighbor.neighborB.begin() + end,
                                           res.begin(), [&](uint32_t pIdx) {
                                             auto const p = pointA[pIdx];
                                             auto const diff =
                                                 computeDiff(p, q);
                                             auto const tolerance = 1.0e-7;
                                             return diff > tolerance;
                                           });
          res.resize(ite - res.begin());
          thrust::sort(res.begin(), res.end());
          return res;
        };
        auto const lNeighbor = filterNeighbor(l), rNeighbor = filterNeighbor(r);
        if (lNeighbor.size() == rNeighbor.size()) {
          for (auto i = decltype(lNeighbor.size())(0); i < lNeighbor.size();
               ++i) {
            assert(lNeighbor[i] == rNeighbor[i]);
          }
        } else {
          std::cout << "l neighbor:\n";
          for (auto pIdx : lNeighbor) {
            auto const p = pointA[pIdx];
            auto const diff = computeDiff(p, q);
            std::cout << p.format(CommaInitFmt)
                      << ", distance: " << std::setprecision(17) << diff
                      << '\n';
          }
          std::cout << "r neighbor:\n";
          for (auto pIdx : rNeighbor) {
            auto const p = pointA[pIdx];
            auto const diff = computeDiff(p, q);
            std::cout << p.format(CommaInitFmt)
                      << ", distance: " << std::setprecision(17) << diff
                      << '\n';
          }
          assert(false);
        }
      });
}
