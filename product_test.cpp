#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;
using Element = float;

int main() {
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
  TiledCopy tiled_copy = make_tiled_copy(Atom{},             // Access strategy
                                         thr_layout,         // thread layout (e.g. 32x4 Col-Major)
                                         val_layout);        // value layout (e.g. 4x1)

  /*TiledCopy vector_copy = make_tiled_copy(AutoVectorizingCopy {},// Access strategy
                                         thr_layout,         // thread layout (e.g. 32x4 Col-Major)
                                         val_layout);        // value layout (e.g. 4x1)
                                                             */
  return 0;
}
