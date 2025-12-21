
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  using Element = float;
  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(256, 512);
  // Define a statically sized block (M, N).
  // Note, by convention, capital letters are used to represent static modes.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  auto tensor_T = tiled_divide(make_layout(tensor_shape), block_shape);// ((M, N), m', n')
  std::cout<< "Tiled divide    " << make_layout(tensor_shape) << " by " << block_shape
    << "-->" << tensor_T << std::endl;

  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

  // Value arrangement per thread
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx

  // Define `AccessType` which controls the size of the actual memory access instruction.
  using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;     // A very specific access width copy instruction
  //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
  //using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<uint_byte_t<sizeof(Element) * size(val_layout)>>::value>>>;

  std::cout << "SrcLayout:      " << SrcLayout {} << std::endl;
  // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
  using Atom = Copy_Atom<CopyOp, Element>;
  auto layout_mn = raked_product(thr_layout, val_layout);
  std::cout << "logical_product "<< thr_layout <<" and "<< val_layout <<"-->";
  std::cout << logical_product(thr_layout, val_layout) <<std::endl;
  std::cout << "raked_product   "<< thr_layout <<" and "<< val_layout <<"-->";
  std::cout << layout_mn <<std::endl;

  auto inv_layout_mn = right_inverse(layout_mn);
  std::cout << "inv_layout_mn   " << inv_layout_mn << std::endl;
  auto inv_logical_mn = right_inverse(logical_product(thr_layout, val_layout));
  std::cout << "inv_logical_mn  " << inv_logical_mn << std::endl;

  TiledCopy tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);

  ThrCopy thr_copy = tiled_copy.get_thread_slice(3);
  auto S = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)), tensor_T);
  auto D = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)), tensor_T);

    // Slice the tensors to obtain a view into each tile.
  Tensor tile_S = S(make_coord(_, _), 0, 0);  // (BlockShape_M, BlockShape_N)
  std::cout << "Tile_S:         " << tile_S.layout() << std::endl;
  Tensor tile_D = D(make_coord(_, _), 1, 1);  // (BlockShape_M, BlockShape_N)
  std::cout << "Tile_D:         " << tile_D.layout() << std::endl;

  auto tiler = product_each(shape(layout_mn));

  Tensor D_S = zipped_divide(tile_S, tiler);
  std::cout << "Divided_S:      " << D_S.layout() << std::endl;
  Tensor D_D = zipped_divide(tile_D, tiler);
  std::cout << "Divided_D:      " << D_D.layout() << std::endl;

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
  std::cout << "thr_tile_S:     " << thr_tile_S.layout() << std::endl;

  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)
  std::cout << "thr_tile_D:     " << thr_tile_D.layout() << std::endl;

  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)
  std::cout << "frament:        " << fragment.layout() << std::endl;

  auto inv_atomlayout = right_inverse(decltype(tiled_copy)::AtomLayoutRef {});
  std::cout << "inv_atomlayout  " << inv_atomlayout << std::endl;
  auto composed_layout = inv_atomlayout.compose(decltype(tiled_copy)::AtomLayoutSrc {});
  std::cout << "composed_layout " << composed_layout << std::endl;
}
