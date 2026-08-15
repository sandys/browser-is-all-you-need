#pragma once

#include <mutex>

namespace workshop_tokens {

class token_box {
public:
    token_box();
    void unlock();
    void add_tokens(int amount);
    void take_tokens(int amount);
    void lock();
    int token_count();

private:
    int count_;
    bool unlocked_;
    std::mutex mutex_;
};

}  // namespace workshop_tokens
