#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace montestrike {

// Cross-platform aligned memory management
class AlignedMemory {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 32;  // 32-byte alignment for AVX2 and cache optimization
    
    // Allocate aligned memory
    static void* allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT);
    
    // Free aligned memory
    static void deallocate(void* ptr);
    
    // Check if pointer is aligned
    static bool is_aligned(const void* ptr, size_t alignment = DEFAULT_ALIGNMENT);
    
    // Align size to next boundary
    static size_t align_size(size_t size, size_t alignment = DEFAULT_ALIGNMENT);
};

// RAII wrapper for aligned memory arrays
template<typename T>
class AlignedArray {
private:
    T* data_;
    size_t size_;
    size_t alignment_;
    
public:
    explicit AlignedArray(size_t count, size_t alignment = AlignedMemory::DEFAULT_ALIGNMENT)
        : data_(nullptr), size_(count), alignment_(alignment) {
        if (count > 0) {
            data_ = static_cast<T*>(AlignedMemory::allocate(count * sizeof(T), alignment));
            if (!data_) {
                throw std::bad_alloc();
            }
        }
    }
    
    ~AlignedArray() {
        if (data_) {
            AlignedMemory::deallocate(data_);
        }
    }
    
    // Non-copyable, movable
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;
    
    AlignedArray(AlignedArray&& other) noexcept 
        : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            if (data_) {
                AlignedMemory::deallocate(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            alignment_ = other.alignment_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Access
    T* get() { return data_; }
    const T* get() const { return data_; }
    
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    size_t size() const { return size_; }
    size_t alignment() const { return alignment_; }
    
    bool is_aligned() const { 
        return AlignedMemory::is_aligned(data_, alignment_); 
    }
    
    // Iterator support
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
};

// Convenience typedefs for common use cases
using AlignedFloatArray = AlignedArray<float>;
using AlignedDoubleArray = AlignedArray<double>;
using AlignedIntArray = AlignedArray<int32_t>;
using AlignedUIntArray = AlignedArray<uint32_t>;

} // namespace montestrike