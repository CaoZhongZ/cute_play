
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

template <class LayoutAtom, class MMATileShape, class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr
auto
tile_to_mma_shape(LayoutAtom const& atom, MMATileShape const& mma_tile_shape, ModeOrder const& order = {})
{
  constexpr int R = decltype(rank(mma_tile_shape))::value;
  auto mn_shape = cute::tuple_cat(zip(shape<0>(mma_tile_shape), take<1,3>(mma_tile_shape)), take<3,R>(mma_tile_shape));
  auto mn_tiled = tile_to_shape(atom, mn_shape, order);                      // (BLK_M,BLK_N,...)
  return tiled_divide(mn_tiled, product_each(shape<0>(mma_tile_shape)));     // ((MMA_M,MMA_N),M_MMAs,N_MMAs,...)
}

int main() {
  auto t_tile = make_tile(Int<128>{}, Int<256>{});
  std::cout << t_tile << std::endl;

  auto t_layout = make_layout(t_tile);

  std::cout << t_layout(_) << std::endl;

  auto glayout = make_layout(Shape(1024, 1024), Stride(1024, 1));
  std::cout << "divide 1024,1024 to 128, 256: " << logical_divide(glayout, t_tile)
    << std::endl;

  auto mma_shape_A = make_shape(make_shape(Int<128>{}, Int<128>{}), _1{}, _4{});
  std::cout << "mma_shape_A: " << mma_shape_A << std::endl;

  /*using Layout_K_SW128_Atom_Bits =
    ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
  using Layout_K_SW128_Atom = decltype(upcast<16>(Layout_K_SW128_Atom_Bits{}));*/
  // std::cout << "Layout SW128: " << Layout_K_SW128_Atom {} << std::endl;
  using Layout_normal = Layout<Shape<_8,_512>,Stride<_512,_1>>;

  // auto sA_layout = tile_to_mma_shape(Layout_K_SW128_Atom {}, mma_shape_A);
  // std::cout << "sA_layout: " << sA_layout << std::endl;

  std::cout << size(Int<1>{}) << std::endl;

  auto sA_layout1 = tile_to_mma_shape(Layout_normal {}, mma_shape_A);
  std::cout << "sA_layout1: " << sA_layout1 << std::endl;

  auto cta_v_tile = make_identity_layout(Shape(1024, 1024)).compose(Shape(128, 128));
  std::cout << "cta_v_tile: " << cta_v_tile << std::endl;

  auto inv_sA_layout = right_inverse(sA_layout1);
  std::cout << "Invert sA_layout1: " << inv_sA_layout << std::endl;

  auto composed = composition(cta_v_tile, inv_sA_layout);
  std::cout << "Compose cta_v_tile and inv_sA_layout: " << composed << std::endl;
  std::cout << "Coalesce composed results: " << coalesce(composed) << std::endl;
}
