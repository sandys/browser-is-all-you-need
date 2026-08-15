#pragma once

#include <mutex>

namespace prepaid_data {

class data_wallet {
public:
    data_wallet() = default;
    void connect();
    void add_megabytes(int amount);
    void use_megabytes(int amount);
    void disconnect();
    int remaining_megabytes();

private:
    int value_{0};
    bool active_{false};
    std::mutex mutex_{};
};

}  // namespace prepaid_data
