
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  // smem_layout: (_32,(_2,_4)):(_2,(_1,_64))
  auto smem_layout = Layout<Shape <_32,Shape <_2, _2>>,
                           Stride< _2,Stride<_1,_64>>>{};
  // auto smem_layout = Layout<Shape <_32, _24>,
  //                          Stride< _24, _1>>{};
  /*
    tiled_copy: TiledCopy
      Tiler_MN:       (_32,_8)
      TiledLayout_TV: (_32,_8):(_1,_32)
    Copy_Atom
      ThrID:        _32:_1
      ValLayoutSrc: ((_16,_2),_8):((_8,_0),_1)
      ValLayoutDst: (_32,(_2,_2)):(_2,(_1,_64))
      ValLayoutRef: (_32,(_2,_2)):(_2,(_1,_64))
      ValueType:    16b
   */
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x2_LDSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_4>>{});

  /* auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});*/

  print("smem_layout: ");print(smem_layout); print("\n");
  print(tiled_copy); print("\n");

  uint16_t gmem[size(smem_layout)];
  uint16_t smem[size(smem_layout)];

  auto t_g_in  = make_tensor(make_gmem_ptr((uint16_t *)gmem), smem_layout);
  auto t_g_out = make_tensor(make_gmem_ptr((uint16_t *)gmem), smem_layout);
  auto t_smem  = make_tensor(make_smem_ptr((uint16_t *)smem), smem_layout);

  volatile int thread = 4;
  /*
    thr_copy: ThrCopy
      ThrIdx: 4
    TiledCopy
      Tiler_MN:       (_32,_8)
      TiledLayout_TV: (_32,_8):(_1,_32)
    Copy_Atom
      ThrID:        _32:_1
      ValLayoutSrc: ((_16,_2),_8):((_8,_0),_1)
      ValLayoutDst: (_32,(_2,_2)):(_2,(_1,_64))
      ValLayoutRef: (_32,(_2,_2)):(_2,(_1,_64))
      ValueType:    16b
   */
  auto thr_copy = tiled_copy.get_thread_slice(thread);
  print("thr_copy: "); print(thr_copy); print("\n");

  auto thr_copy2 = tiled_copy.get_thread_slice(thread + 1);

  auto tXsX = thr_copy.partition_S(t_smem);   // (V,M,N)
  auto tXsX2 = thr_copy2.partition_S(t_smem);

  auto tXgX = thr_copy.partition_D(t_g_out);  // (V,M,N)

  auto tXrX = make_tensor<uint16_t>(shape(tXgX)); // (V,M,N)

  printf("gmem %p, smem %p\n", gmem, smem);

  print("tXsX: " ); print(tXsX); print("\n");
  print("tXsX2: " ); print(tXsX2); print("\n");
  print("tXgX: " ); print(tXgX); print("\n");
  print("tXrX: " ); print(tXrX); print("\n");

  // Copy smem -> rmem via tiled_copy (LDSM, LDS)
  copy(tiled_copy, tXsX, tXrX);

  // Output rmem -> gmem
  copy(tXrX, tXgX);
}
