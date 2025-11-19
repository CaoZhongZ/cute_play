CXX=clang++
CXXFLAGS=-g3 -O0 -ggdb3 -fno-inline -fno-omit-frame-pointer --std=c++17 -Icutlass/include -Icutlass/tools/util/include -Iinclude

all : coalesce_test composition_test complement_test tiling_test

clean:
	rm -rf coalesce_test composition_test complement_test tiling_test *.dSYM
