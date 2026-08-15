#include "prepaid_data.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class check_suite {
public:
    void expect(bool condition, std::string_view description) {
        ++count_;
        if (!condition) {
            failed_ = true;
            std::cerr << "FAILED: " << description << '\n';
        }
    }

    template <typename Function>
    void expect_runtime_error(Function&& function, std::string_view description) {
        ++count_;
        try {
            std::forward<Function>(function)();
        } catch (std::runtime_error const&) {
            return;
        } catch (std::exception const& error) {
            failed_ = true;
            std::cerr << "FAILED: " << description
                      << " threw a different exception: " << error.what() << '\n';
            return;
        }
        failed_ = true;
        std::cerr << "FAILED: " << description
                  << " did not throw std::runtime_error\n";
    }

    std::size_t count() const { return count_; }
    bool failed() const { return failed_; }

private:
    std::size_t count_{0};
    bool failed_{false};
};

}  // namespace

int main() {
    check_suite checks;

    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        checks.expect(subject.remaining_megabytes() == 0, "newly_started_zero");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(100);
        checks.expect(subject.remaining_megabytes() == 100, "single_credit");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(100);
        subject.add_megabytes(50);
        checks.expect(subject.remaining_megabytes() == 150, "multiple_credits");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(100);
        subject.use_megabytes(75);
        checks.expect(subject.remaining_megabytes() == 25, "single_debit");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(100);
        subject.use_megabytes(80);
        subject.use_megabytes(20);
        checks.expect(subject.remaining_megabytes() == 0, "multiple_debits");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(100);
        subject.add_megabytes(110);
        subject.use_megabytes(200);
        subject.add_megabytes(60);
        subject.use_megabytes(50);
        checks.expect(subject.remaining_megabytes() == 20, "sequential_operations");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.disconnect();
        checks.expect_runtime_error(
            [&]() { (void)subject.remaining_megabytes(); },
            "value_after_stop_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.disconnect();
        checks.expect_runtime_error(
            [&]() { subject.add_megabytes(50); },
            "credit_after_stop_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        checks.expect_runtime_error(
            [&]() { subject.add_megabytes(50); },
            "credit_before_start_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.disconnect();
        checks.expect_runtime_error(
            [&]() { subject.use_megabytes(50); },
            "debit_after_stop_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        checks.expect_runtime_error(
            [&]() { subject.disconnect(); },
            "stop_before_start_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        checks.expect_runtime_error(
            [&]() { subject.connect(); },
            "start_twice_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(50);
        subject.disconnect();
        subject.connect();
        checks.expect(subject.remaining_megabytes() == 0, "restart_resets_zero");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(25);
        checks.expect_runtime_error(
            [&]() { subject.use_megabytes(50); },
            "overdraft_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        subject.add_megabytes(100);
        checks.expect_runtime_error(
            [&]() { subject.use_megabytes(-50); },
            "negative_debit_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        checks.expect_runtime_error(
            [&]() { subject.add_megabytes(-50); },
            "negative_credit_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.add_megabytes(1);
                std::this_thread::sleep_for(5ms);
                subject.use_megabytes(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.remaining_megabytes() == 0, "concurrent_transactions");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        checks.expect_runtime_error(
            [&]() { subject.use_megabytes(0); },
            "zero_debit_throws");
    }
    {
        prepaid_data::data_wallet subject{};
        subject.connect();
        checks.expect_runtime_error(
            [&]() { subject.add_megabytes(0); },
            "zero_credit_throws");
    }

    if (checks.failed() || checks.count() != 19U) {
        std::cerr << "Verification failed after " << checks.count()
                  << " assertions\n";
        return 1;
    }

    std::cout << "All tests passed (19 assertions in 19 test cases)\n";
    return 0;
}
