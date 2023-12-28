#pragma once
#include <array>
#include <cstdint>

namespace multirange {
template <std::size_t dim>
using MultiRange = std::array<std::tuple<int32_t, int32_t>, dim>;

template <std::size_t dim, typename Op>
auto forInMultiRange(MultiRange<dim> const &multiRange, Op const &op) -> void;

template <std::size_t dim, typename T>
auto constexpr constructArray(T v) -> std::array<T, dim>;
} // namespace multirange

namespace multirange {
namespace _detail {
template <std::size_t dim, typename T, std::size_t depth>
auto constexpr _implementConsturctArray(T v, std::array<T, dim> arr)
    -> std::array<T, dim> {
  if constexpr (depth < dim) {
    arr[dim - 1 - depth] = v;
    return _implementConsturctArray<dim, T, depth + 1>(v, arr);
  } else {
    return arr;
  }
}

template <std::size_t dim, typename Op, std::size_t depth>
auto _implementForInMultiRange(MultiRange<dim> const &multiRange, Op const &op,
                               std::array<int32_t, dim> pos) -> void {
  if constexpr (depth < dim) {
    auto constexpr recurDim = dim - 1 - depth;
    auto const [start, end] = multiRange[recurDim];
    for (auto i = start; i < end; ++i) {
      pos[recurDim] = i;
      _implementForInMultiRange<dim, Op, depth + 1>(multiRange, op, pos);
    }
  } else {
    op(pos);
  }
}

} // namespace _detail

template <std::size_t dim, typename Op>
auto forInMultiRange(MultiRange<dim> const &multiRange, Op const &op) -> void {
  _detail::_implementForInMultiRange<dim, Op, 0>(multiRange, op,
                                                 std::array<int32_t, dim>());
}
template <std::size_t dim, typename T>
auto constexpr constructArray(T v) -> std::array<T, dim> {
  return _detail::_implementConsturctArray<dim, T, 0>(v, std::array<T, dim>());
}

} // namespace multirange
