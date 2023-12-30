#pragma once
#include "compute_neighbor.cuh"
#include <iostream>
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

auto inline checkNeighbor(NeighborsHost const &l) -> void {
  assert(l.offsetA.size() == l.counterA.size());
}

auto inline checkNeighborConsistent(NeighborsHost const &l,
                                    NeighborsHost const &r) -> void {
  checkNeighbor(l);
  checkNeighbor(r);

  std::cout << l.offsetA.size() << ' ' << r.offsetA.size() << std::endl;
  assert(l.offsetA.size() == r.offsetA.size());
  std::cout << l.neighborB.size() << ' ' << r.neighborB.size() << std::endl;
  assert(l.neighborB.size() == r.neighborB.size());

  auto const a = l.offsetA.size();

  auto lNeighborB = l.neighborB, rNeighborB = r.neighborB;

  thrust::for_each(
      thrust::host, thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(a), [&](uint32_t idx) {
        assert(l.offsetA[idx] == r.offsetA[idx]);
        assert(l.counterA[idx] == r.counterA[idx]);
        auto const start = l.offsetA[idx], end = start + l.counterA[idx];
        std::sort(lNeighborB.begin() + start, lNeighborB.begin() + end);
        std::sort(rNeighborB.begin() + start, rNeighborB.begin() + end);
        for (auto i = start; i < end; ++i) {
          assert(lNeighborB[i] == rNeighborB[i]);
        }
      });
}
