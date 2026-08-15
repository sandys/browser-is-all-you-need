#include "arcade_card.h"

#include <stdexcept>

namespace arcade_card {

player_card::player_card() = default;

void player_card::issue() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ == true) {
        throw std::runtime_error("resource is already active");
    }
    value_ = 0;
    active_ = true;
}

void player_card::load(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    require_active();
    require_positive(amount);
    value_ += amount;
}

void player_card::spend(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    require_active();
    require_positive(amount);
    if (amount > value_) {
        throw std::runtime_error("amount exceeds available value");
    }
    value_ -= amount;
}

void player_card::revoke() {
    std::lock_guard<std::mutex> guard(mutex_);
    require_active();
    active_ = false;
}

int player_card::credits() {
    std::lock_guard<std::mutex> guard(mutex_);
    require_active();
    return value_;
}

void player_card::require_active() const {
    if (!active_) {
        throw std::runtime_error("resource is not active");
    }
}

void player_card::require_positive(int amount) {
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
}

}  // namespace arcade_card
