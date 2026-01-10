
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <iostream>

using namespace cute;

template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {

  using Blk_MN    = _128;
  using Blk_SF    =   _4; 

  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>, 
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>, 
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;
  using SfAtom    = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};

//// Describe the Scalefactor Tensor without VectorSize
struct Sm1xxBlockScaledTensorConfig {
  // k-major order
  // The blockscaled tensor does not need to know vectorsize
  using Blk_M = _128;
  using Blk_N =   _4; 
  using SfAtom = Layout< Shape< Shape<_32,_4>,  Shape<_4>>, 
                        Stride<Stride<_16,_4>, Stride<_1>>>;

  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape(ProblemShape problem_shape) {
    auto problem_shape_MNL = append<3>(problem_shape, 1);
    auto [M, N, L] = problem_shape_MNL;
    return tile_to_shape(SfAtom{}, make_shape(M,N,L), Step<_2,_1,_3>{});
  }
};

int main() {
  static constexpr int SFVecSize = 16;
  using Sm1xxBlkScaledChunk = Sm1xxBlockScaledBasicChunk<SFVecSize>;
  using Blk_MN = typename Sm1xxBlkScaledChunk::Blk_MN;
  using Blk_SF = typename Sm1xxBlkScaledChunk::Blk_SF; 
  using SfAtom = typename Sm1xxBlkScaledChunk::SfAtom;

  using LayoutSF = decltype(blocked_product(SfAtom{}, make_layout( make_shape(int32_t(0), int32_t(0), int32_t(0)),
                                                                  make_stride(_1{}, int32_t(0), int32_t(0)))));


  print("SfAtom:  \t"); print(SfAtom {}); print("\n");
  print("LayoutSF:\t"); print(LayoutSF {}); print("\n");

  auto problem_shape_MNKL = make_shape(256, 512, 1024, 1);
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
  const auto layout_sfa_ref = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
  const auto layout_sfb_ref = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

  print("\nproblem_shape_MNKL:\t"); print(problem_shape_MNKL); print("\n");
  print("layout_sfa_ref:\t"); print(layout_sfa_ref); print("\n");
  print("layout_sfb_ref:\t"); print(layout_sfb_ref); print("\n");
}
