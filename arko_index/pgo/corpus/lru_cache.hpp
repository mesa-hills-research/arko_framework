// Training sample for the arko_index PGO profiles: an LRU cache.
// Written for this purpose; it is parsed, never compiled.
#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#define ARKO_LIKELY(x) __builtin_expect(!!(x), 1)
#ifndef ARKO_CACHE_DEFAULT_CAPACITY
#define ARKO_CACHE_DEFAULT_CAPACITY 128
#endif

namespace arko::sample {

struct Stats {
    std::size_t hits = 0;
    std::size_t misses = 0;
    double hit_ratio() const noexcept {
        const auto total = hits + misses;
        return total == 0 ? 0.0 : static_cast<double>(hits) / total;
    }
};

template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache {
public:
    using key_type = Key;
    using value_type = Value;
    using Evicted = std::function<void(const Key&, Value&&)>;

    explicit LruCache(std::size_t capacity = ARKO_CACHE_DEFAULT_CAPACITY, Evicted on_evict = {})
        : capacity_(capacity == 0 ? 1 : capacity), on_evict_(std::move(on_evict)) {}

    LruCache(const LruCache&) = delete;
    LruCache& operator=(const LruCache&) = delete;
    LruCache(LruCache&&) noexcept = default;
    ~LruCache() = default;

    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = index_.find(key);
        if (ARKO_LIKELY(it != index_.end())) {
            order_.splice(order_.begin(), order_, it->second);
            ++stats_.hits;
            return it->second->second;
        }
        ++stats_.misses;
        return std::nullopt;
    }

    template <typename V>
    void put(const Key& key, V&& value) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (auto it = index_.find(key); it != index_.end()) {
            it->second->second = std::forward<V>(value);
            order_.splice(order_.begin(), order_, it->second);
            return;
        }
        order_.emplace_front(key, std::forward<V>(value));
        index_[key] = order_.begin();
        while (order_.size() > capacity_) {
            auto& last = order_.back();
            if (on_evict_) {
                on_evict_(last.first, std::move(last.second));
            }
            index_.erase(last.first);
            order_.pop_back();
        }
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return order_.size();
    }

    Stats stats() const { return stats_; }

    template <typename F>
    void for_each(F&& visit) const {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const auto& [key, value] : order_) {
            visit(key, value);
        }
    }

private:
    using Node = std::pair<Key, Value>;
    std::size_t capacity_;
    Evicted on_evict_;
    std::list<Node> order_;
    std::unordered_map<Key, typename std::list<Node>::iterator, Hash> index_;
    mutable std::mutex mutex_;
    Stats stats_{};
};

inline std::string describe(const Stats& s) {
    return "hits=" + std::to_string(s.hits) + " misses=" + std::to_string(s.misses);
}

}  // namespace arko::sample
