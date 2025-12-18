
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
  auto tensor_S = tiled_divide(make_layout(tensor_shape), block_shape);// ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

  // Value arrangement per thread
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx
  // Define `AccessType` which controls the size of the actual memory access instruction.
  using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;     // A very specific access width copy instruction
  //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
  //using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

  // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
  using Atom = Copy_Atom<CopyOp, Element>;
  TiledCopy tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);
}
