
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <iostream>

#include <cutlass/detail/layout.hpp>

using namespace cute;

// The shared memory buffers for A and B matrices.
template <class TypeA,           // Tensor A data type
          class TypeB,           // Tensor B data type
          class ASmemLayout,     // (MmaA, NumMma_M, NumMma_K, ...)
          class BSmemLayout>     // (MmaB, NumMma_N, NumMma_K, ...)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier;   // Barrier to track MMA computation on SMEM

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

namespace cutlass::gemm::collective {
  struct KernelScheduleAuto {};
}

namespace cute::detail {
namespace blockscaled {

enum class BlockScaledInstr {
  MXF4_NVF4,
  MXF4F6F8
};

}}

template <class BuilderScheduleTag, class T>
struct blockscaled_type {};

template <class BuilderScheduleTag, class T>
struct blockscaled_type<BuilderScheduleTag, cutlass::nv_float4_t<T>> {
  using sf_type = cutlass::float_ue4m3_t;
  using data_type = T;
  static constexpr uint32_t SfVectorSize = 16;
};

template <class Element, bool IsF8F6F4 = true>
constexpr auto
sm1xx_kernel_input_element_to_mma_input_element() {
  if constexpr (cute::is_same_v<Element, float>) {
    return cutlass::tfloat32_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::float_e2m1_t> && IsF8F6F4) {
    return cutlass::detail::float_e2m1_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::float_e3m2_t> && IsF8F6F4) {
    return cutlass::detail::float_e3m2_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::float_e2m3_t> && IsF8F6F4) {
    return cutlass::detail::float_e2m3_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::type_erased_dynamic_float4_t> && IsF8F6F4) {
    return cutlass::detail::type_erased_dynamic_float4_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::type_erased_dynamic_float6_t> && IsF8F6F4) {
    return cutlass::detail::type_erased_dynamic_float6_unpacksmem_t{};
  }
  else {
    return Element{};
  }
}

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

  using TypeA = cutlass::nv_float4_t<cutlass::float_e2m1_t>; // MMA A Data Type
  using TypeB = cutlass::nv_float4_t<cutlass::float_e2m1_t>; // MMA B Data Type
  using TypeC = cutlass::bfloat16_t;   // MMA C Data Type
  using TypeD = cutlass::bfloat16_t;   // MMA D Data Type
  using TypeAccumulator = float;

  using LayoutATag  = cutlass::layout::RowMajor;
  using LayoutBTag  = cutlass::layout::RowMajor;
  using LayoutCTag  = cutlass::layout::RowMajor;  // Layout type for C matrix operand
  using LayoutDTag  = cutlass::layout::RowMajor;  // Layout type for D matrix operand

  using ScheduleTag = cutlass::gemm::collective::KernelScheduleAuto;

  using MmaTileShape = Shape<_128,_256,_256>;   // MMA's tile size
  using ClusterShape = Shape<_1,_4,_1>;    // Shape of the threadblocks in a cluster
                                           // (_2, _4, _1) is for 2SM variant

  using ElementA = blockscaled_type<ScheduleTag, TypeA>::data_type;
  using ElementSFA = blockscaled_type<ScheduleTag, TypeA>::sf_type;
  using ElementB = blockscaled_type<ScheduleTag, TypeB>::data_type;
  using ElementSFB = blockscaled_type<ScheduleTag, TypeB>::sf_type;
  using ElementSF = ElementSFA;

  using ElementAMma = decltype(
      sm1xx_kernel_input_element_to_mma_input_element<ElementA, false>());
  using ElementBMma = decltype(
      sm1xx_kernel_input_element_to_mma_input_element<ElementB, false>());

  // Create TiledMma. make_tiled_mma takes the target instructions and an (optional) instruction layout as parameters to create a
  // larger TiledMma from the given mma instruction.
  // See cute/arch/mma_sm100_umma.hpp for all tcgen05.mma instructions
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_MXF4_SS<ElementAMma, ElementBMma, TypeAccumulator,
      ElementSF,                        // Mma's A, B, and Accumulator types
      128, 256, 16,                     // Mma M and N dimensions and vec size
      UMMA::Major::K, UMMA::Major::K>{} // A and B layouts
  );

  print(tiled_mma);

  // Define MMA tiler sizes (static)
  auto bM = tile_size<0>(tiled_mma);    // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
  auto bN = tile_size<1>(tiled_mma);    // MMA Tile N. We'll use 1 MMAs per MMA Tile M.
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For 16b types, tcgen05.mma has K16.
  auto mma_tiler = make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)
  std::cout << "mma_tiler:\t" << mma_tiler << std::endl;
  auto sf_tiler = make_shape(bM, bN, bK/16);       // (MMA_M, MMA_N, MMA_K)

  if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
    std::cerr << "The MMA Shape should evenly divide the MMA Tiler." << std::endl;
    return -1;
  }

  /*if (not evenly_divides(make_shape(Gemm_M, Gemm_N, Gemm_K), mma_tiler)) {
    std::cerr << "OOB accesses are not supported. MmaTiler_MNK should evenly divide ProblemShape_MNK." << std::endl;
    return -1;
  }*/
  // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned (MmaA, NumMma_M, NumMma_K)
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler), _4{}));
  // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned (MmaB, NumMma_N, NumMma_K)
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler), _4{}));

  // Print and inspect mma_shape_A, and mma_shape_B for this example.
  print("mma_shape_A:\t"); print(mma_shape_A); print("\n");  // mma_shape_A:  ((_128,_16),_1,_4)
  print("mma_shape_B:\t"); print(mma_shape_B); print("\n");  // mma_shape_B:  ((_256,_16),_1,_4)

  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<ElementA>{}, mma_shape_A, Step<_1,_2,_3>{});
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<ElementB>{}, mma_shape_B, Step<_1,_2,_3>{});

  std::cout<<"UMMA::Layout_K_SW128_Atom:\t"<<UMMA::Layout_K_SW128_Atom<ElementA>{}<<std::endl;

  // Print and inspect sA_layout and sB_layout for this example.
  print("sA_layout:\t"); print(sA_layout); print("\n");      // sA_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_64),_1,_4):((_256,_1),_0,_64)
  print("sB_layout:\t"); print(sB_layout); print("\n");      // sB_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_64),_1,_4):((_256,_1),_0,_64)

  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  auto layout_sfa = Sm1xxBlkScaledConfig::deduce_layoutSFA();
  auto layout_sfb = Sm1xxBlkScaledConfig::deduce_layoutSFB();

  print("layout_sfa and layout_sfb is used for global SF description\n");
  print("layout_sfa:\t"); print(layout_sfa); print("\n");
  print("layout_sfb:\t"); print(layout_sfb); print("\n");

  auto sfA_layout = Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(tiled_mma, MmaTileShape{});
  auto sfB_layout = Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(tiled_mma, MmaTileShape{});

  print("sfA_layout:\t"); print(sfA_layout); print("\n");
  print("sfB_layout:\t"); print(sfB_layout); print("\n");

  auto smem_sfA_layout = make_layout(
    append(shape(sfA_layout), Int<4>{}),
    append(stride(sfA_layout), size(filter_zeros(sfA_layout)))
  );

  print("smem_sfA_laylout:\t"); print(smem_sfA_layout); print("\n");

  auto smem_sfB_layout = make_layout(
    append(shape(sfB_layout), Int<4>{}),
    append(stride(sfB_layout), size(filter_zeros(sfA_layout)))
  );

  print("smem_sfB_laylout:\t"); print(smem_sfB_layout); print("\n");

  // The cluster shape and layout
  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = make_coord(0 % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                                   1 / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                                   1,                                //    MMA-N coordinate
                                   _);                                        //    MMA-K coordinate

  int Gemm_M = 512;
  int Gemm_N = 1024;
  int Gemm_K = 2048;

  // A tensor MxK K-major (Layout T = Row-Major)
  Layout layout_A = make_layout(make_shape (Gemm_M,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
  // B tensor NxK K-major (Layout N = Column-Major)
  Layout layout_B = make_layout(make_shape (Gemm_N,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
  // C tensor MxN N-major (Layout T = Row-Major)
  Layout layout_C = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  // D tensor MxN N-major (Layout T = Row-Major)
  Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
                                                                  //
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(Gemm_M, Gemm_N, Gemm_K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(Gemm_M, Gemm_N, Gemm_K, 1));

  print("layout_SFA:\t"); print(layout_SFA); print("\n");
  print("layout_SFB:\t"); print(layout_SFB); print("\n");

  Tensor mA = make_tensor(make_gmem_ptr(static_cast<ElementA *>(nullptr)), layout_A);      // (Gemm_M, Gemm_K)
  Tensor mB = make_tensor(make_gmem_ptr(static_cast<ElementB *>(nullptr)), layout_B);      // (Gemm_N, Gemm_K)

  Tensor msfA = make_tensor(make_gmem_ptr(static_cast<ElementSFA *>(nullptr)), layout_SFA);
  Tensor msfB = make_tensor(make_gmem_ptr(static_cast<ElementSFA *>(nullptr)), layout_SFB);

  Tensor mC = make_tensor(make_gmem_ptr(static_cast<TypeC *>(nullptr)), layout_C);      // (Gemm_M, Gemm_N)
  Tensor mD = make_tensor(make_gmem_ptr(static_cast<TypeC *>(nullptr)), layout_D);      // (Gemm_M, Gemm_N)

  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  print("Index with:\t"); print(mma_coord); print("\n");

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});  // (MmaTile_M, MmaTile_K, Tiles_K)
  Tensor gsfA = local_tile(msfA, sf_tiler, mma_coord, Step<_1, X,_1>{});

  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});  // (MmaTile_N, MmaTile_K, Tiles_K)
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)

  print("mA:\t"); print(mA); print("\n");   // mA:   gmem_ptr[16b](GMEM_ADDR_A) o (512,256):(256,_1)
  print("msfA:\t"); print(msfA); print("\n");   // mA:   gmem_ptr[16b](GMEM_ADDR_A) o (512,256):(256,_1)
  print("mB:\t"); print(mB); print("\n");   // mB:   gmem_ptr[16b](GMEM_ADDR_B) o (1024,256):(256,_1)
  print("mC:\t"); print(mC); print("\n");   // mC:   gmem_ptr[32b](GMEM_ADDR_C) o (512,1024):(1024,_1)
  print("mD:\t"); print(mD); print("\n");   // mD:   gmem_ptr[32b](GMEM_ADDR_D) o (512,1024):(1024,_1)

  print("mA tiled:\t"); print(zipped_divide(mA, dice(Step<_1, X, _1>{}, mma_tiler))); print("\n");
  print("mB tiled:\t"); print(zipped_divide(mB, dice(Step<X, _1, _1>{}, mma_tiler))); print("\n");
  print("mC tiled:\t"); print(zipped_divide(mC, dice(Step<_1, _1, X>{}, mma_tiler))); print("\n");
  print("mD tiled:\t"); print(zipped_divide(mD, dice(Step<_1, _1, X>{}, mma_tiler))); print("\n");

  print("gA:\t"); print(gA); print("\n");   // gA:   gmem_ptr[16b](GMEM_ADDR_A + offset_for_mma_tile) o (_128,_64,4):(256,_1,_64)
  print("gsfA:\t"); print(gsfA); print("\n");

  print("gB:\t"); print(gB); print("\n");   // gB:   gmem_ptr[16b](GMEM_ADDR_B + offset_for_mma_tile) o (_256,_64,4):(_1,256,16384)
  print("gC:\t"); print(gC); print("\n");   // gC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile) o (_128,_256):(256,_1)
  print("gD:\t"); print(gD); print("\n");   // gD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile) o (_128,_256):(256,_1)

  auto mma_v = get<0>(mma_coord_vmnk);
  print("\nmma_v:\t"); print(mma_v); print("\n");
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // Use Peer CTA coordinate

  print("ThrMMA:\t"); print(cta_mma); print("\n");

  Tensor tCgA = cta_mma.partition_A(gA);         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCgB = cta_mma.partition_B(gB);         // (MmaB, NumMma_N, NumMma_K, Tiles_K)
  Tensor tCgC = cta_mma.partition_C(gC);         // (MmaC, NumMma_M, NumMma_N)
  Tensor tCgD = cta_mma.partition_C(gD);         // (MmaC, NumMma_M, NumMma_N)

  print("tCgA:\t"); print(tCgA); print("\n");  // tCgA:   gmem_ptr[16b](GMEM_ADDR_A + offset_for_mma_tile + offset_for_mma) o ((_128,_16),_1,_4,4):((256,_1),_0,_16,_64)
  print("tCgB:\t"); print(tCgB); print("\n");  // tCgB:   gmem_ptr[16b](GMEM_ADDR_B + offset_for_mma_tile + offset_for_mma) o ((_256,_16),_1,_4,4):((_1,256),_0,4096,16384)
  print("tCgC:\t"); print(tCgC); print("\n");  // tCgC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((256,_1),_0,_0)
  print("tCgD:\t"); print(tCgD); print("\n");  // tCgD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((256,_1),_0,_0)

  using SMEMStorage = SharedStorage<ElementA, ElementB, decltype(sA_layout), decltype(sB_layout)>;
  SMEMStorage& shared_storage = *reinterpret_cast<SMEMStorage*>(static_cast<char *>(nullptr));

  // Represent the SMEM buffers for A and B
  Tensor tCsA = shared_storage.tensor_sA();         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCsB = shared_storage.tensor_sB();         // (MmaB, NumMma_M, NumMma_K, Tiles_K)

  // MMA Fragment Allocation
  // We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm operations.
  // For tcgen05.mma operations:
  // - Matrices A and B are sourced from SMEM
  // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
  // - The first mode of each descriptor represents the SMEM for a single MMA operation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);      // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);      // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  // TMEM Allocation
  // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
  // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the accumulator.
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);    // (MmaC, NumMma_M, NumMma_N)

  print("tCsA:\t"); print(tCsA); print("\n");     // tCsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  print("tCsB:\t"); print(tCsB); print("\n");     // tCsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
  print("tCrA:\t"); print(tCrA); print("\n");     // tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  print("tCrB:\t"); print(tCrB); print("\n");     // tCrB:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  print("tCtAcc:\t"); print(tCtAcc); print("\n"); // tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.

  // TMA load
  auto tma_a = SM90_TMA_LOAD {};
  auto tma_sfa = SM90_TMA_LOAD {};
  auto tma_b = SM90_TMA_LOAD {};
  auto tma_sfb = SM90_TMA_LOAD {};

  print("\ncluster_layout_vmnk"); print(cluster_layout_vmnk); print("\n");
  print("\n-------------------------make_tma_atom_A_sm100----------------------------\n");
  auto tma_a_atom = make_tma_atom_A_sm100<ElementA>(
      tma_a,
      gA,
      sA_layout(_,_,_,Int<0>{}),
      MmaTileShape {},
      tiled_mma,
      cluster_layout_vmnk
    );
  print("tma_a_atom\t"); print(tma_a_atom); print("\n");

  // smem_sfA_laylout:	((((_32,_4),_1),(_16,_4)),_1,(_1,_4),_4):((((_16,_4),_512),(_0,_1)),_0,(_4,_512),_2048)
  // uint16_t <-- not fp8???
  print("\n-------------------------make_tma_atom_A_sm100----------------------------\n");
  auto tma_sfa_atom = make_tma_atom_A_sm100<uint16_t>(
      tma_sfa,
      make_tensor(static_cast<ElementSF const*>(nullptr), layout_SFA),
      smem_sfA_layout(_,_,_,Int<0>{}),
      MmaTileShape {},
      tiled_mma,
      cluster_layout_vmnk
    );
  print("tma_sfa_atom\t"); print(tma_sfa_atom); print("\n");

  print("\nStart mainloop: \n");

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // Step 2a: Load A and B tiles

    // Using auto-vectorized copy operation:
    // - Utilizes 128 threads for parallel data transfer
    // - Copy operations are distributed efficiently across all threads
    // - CuTe can automatically determine optimal vector width
    print("Cooperative_copy:");print(tCgA(_,_,_,k_tile));print(" to ");
    print(tCsA); print("\n");
    print("cooperative_copy:");print(tCgB(_,_,_,k_tile)); print(" to ");
    print(tCsB); print("\n");

    // Step 2b: Execute the MMAs for this tile

    // Wait for loads to SMEM to complete with __syncthreads()
    // __syncthreads();

    // tcgen05.mma instructions require single-thread execution:
    // - Only one warp performs the MMA-related loop operations
    // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
    // - No explicit elect_one_sync region is needed from the user
    // if (elect_one_warp) {
      // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
        // print("gemm(tiled_mma: ");
        // print(tCrA(_,_,k_block)); print(" * ");
        // print(tCrB(_,_,k_block)); print(" -> ");
        // print(tCtAcc);
        // print(")\n");

        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
      // cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
      print("umma_arrive \n");
    // }
    // Wait MMAs to complete to avoid overwriting the A and B SMEM.
    // cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    print("wait_barrier\n");
    mma_barrier_phase_bit ^= 1;
  }
}
