CXX=clang++
CXXFLAGS=-g3 -O0 -ggdb3 -fno-inline -fno-omit-frame-pointer --std=c++17 -Icutlass/include -Icutlass/tools/util/include -Iinclude

all : coalesce_test composition_test complement_test tiling_test product_test

EXCLUDE := clang-format git-helper

clean:
	@for f in *; do \
		echo "$(EXCLUDE)" | grep -qw "$$f" && continue; \
		if [ -f "$$f" ] && file "$$f" | grep -Eq 'ELF .* executable|Mach-O .* executable'; then \
			rm -f "$$f"; \
		fi; \
	done
