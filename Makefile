CXX=clang++
CXXFLAGS=-g3 -O0 -ggdb3 -fno-inline -fno-omit-frame-pointer --std=c++17 -I../cutlass/include -I../cutlass/tools/util/include -Iinclude

all : identity_layout

clean:
	rm -f identity_layout
