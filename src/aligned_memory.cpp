#include "montestrike/aligned_memory.hpp"
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
    #include <malloc.h>
    #include <windows.h>
#else
    #include <cstdlib>
    #include <unistd.h>
    #include <sys/mman.h>
#endif

namespace montestrike {

void* AlignedMemory::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }
    
    // Ensure alignment is a power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    
    void* ptr = nullptr;
    
#ifdef _WIN32
    // Windows: Use _aligned_malloc
    ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
#else
    // POSIX: Use posix_memalign
    int result = posix_memalign(&ptr, alignment, size);
    if (result != 0) {
        throw std::bad_alloc();
    }
#endif
    
    return ptr;
}

void AlignedMemory::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

bool AlignedMemory::is_aligned(const void* ptr, size_t alignment) {
    if (!ptr) {
        return true;  // nullptr is considered aligned
    }
    
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

size_t AlignedMemory::align_size(size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    
    return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace montestrike