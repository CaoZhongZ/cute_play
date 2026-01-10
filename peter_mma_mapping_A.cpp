
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

template <typename ValType, typename LayoutIn>
CUTE_HOST_DEVICE
constexpr auto
wi_interleave(LayoutIn const&)
{
  constexpr LayoutIn layout{};
  constexpr int per_byte = ceil_div(8, sizeof_bits_v<ValType>);
  constexpr int vals = ceil_div(size(layout), _16{});
  auto tv_interleaved = Layout<Shape<_16,          Shape<C<per_byte>, C<vals/per_byte>>>,
                              Stride<C<per_byte>, Stride<_1,          C<_16{} * per_byte>>>>{};
  return coalesce(composition(layout, tv_interleaved), Step<_1,_1>{});
}

template <typename ValType, typename LayoutIn>
using wi_interleave_t = remove_cvref_t<decltype(wi_interleave<ValType>(LayoutIn{}))>;


int main() {
  // Thread arrangement
  using ALayout = wi_interleave_t<uint16_t, Layout<Shape<_16, _8>, Stride<_8, _1>>>;

  auto TV_layout = ALayout {};
  Layout layout_mn = make_identity_layout(make_shape(_8{}, _16{}));

  print(TV_layout); print("\n");
  print(composition(layout_mn, TV_layout)(8, 0)); print("\n");

  print(composition(layout_mn, TV_layout)); print("\n");

  // print_latex_tv(composition(layout_mn, TV_layout), make_tile(_8{}, _16{}));

}
