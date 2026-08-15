#pragma once

#include <mutex>
#include <stdexcept>

namespace inventory_ledger {

class stock_ledger {
public:
    stock_ledger() = default;

    void begin() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (active_) {
            throw std::runtime_error("resource is already active");
        }
        value_ = 0;
        active_ = true;
    }

    void receive(int amount) {
        std::lock_guard<std::mutex> guard(mutex_);
        require_active();
        require_positive(amount);
        value_ += amount;
    }

    void dispatch(int amount) {
        std::lock_guard<std::mutex> guard(mutex_);
        require_active();
        require_positive(amount);
        if (amount > value_) {
            throw std::runtime_error("amount exceeds available value");
        }
        value_ -= amount;
    }

    void end() {
        std::lock_guard<std::mutex> guard(mutex_);
        require_active();
        active_ = false;
    }

    int quantity() {
        std::lock_guard<std::mutex> guard(mutex_);
        require_active();
        return value_;
    }

private:
    void require_active() const {
        if (!active_) {
            throw std::runtime_error("resource is not active");
        }
    }

    static void require_positive(int amount) {
        if (amount <= 0) {
            throw std::runtime_error("amount must be positive");
        }
    }

    int value_{0};
    bool active_{false};
    std::mutex mutex_{};
};

}  // namespace inventory_ledger
