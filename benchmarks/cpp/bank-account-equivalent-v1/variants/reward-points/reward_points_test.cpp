#include "reward_points.h"

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
        reward_points::reward_account subject{};
        subject.open();
        checks.expect(subject.points() == 0, "newly_started_zero");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(100);
        checks.expect(subject.points() == 100, "single_credit");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(100);
        subject.earn(50);
        checks.expect(subject.points() == 150, "multiple_credits");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(100);
        subject.redeem(75);
        checks.expect(subject.points() == 25, "single_debit");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(100);
        subject.redeem(80);
        subject.redeem(20);
        checks.expect(subject.points() == 0, "multiple_debits");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(100);
        subject.earn(110);
        subject.redeem(200);
        subject.earn(60);
        subject.redeem(50);
        checks.expect(subject.points() == 20, "sequential_operations");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.close();
        checks.expect_runtime_error(
            [&]() { (void)subject.points(); },
            "value_after_stop_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.close();
        checks.expect_runtime_error(
            [&]() { subject.earn(50); },
            "credit_after_stop_throws");
    }
    {
        reward_points::reward_account subject{};
        checks.expect_runtime_error(
            [&]() { subject.earn(50); },
            "credit_before_start_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.close();
        checks.expect_runtime_error(
            [&]() { subject.redeem(50); },
            "debit_after_stop_throws");
    }
    {
        reward_points::reward_account subject{};
        checks.expect_runtime_error(
            [&]() { subject.close(); },
            "stop_before_start_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        checks.expect_runtime_error(
            [&]() { subject.open(); },
            "start_twice_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(50);
        subject.close();
        subject.open();
        checks.expect(subject.points() == 0, "restart_resets_zero");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(25);
        checks.expect_runtime_error(
            [&]() { subject.redeem(50); },
            "overdraft_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        subject.earn(100);
        checks.expect_runtime_error(
            [&]() { subject.redeem(-50); },
            "negative_debit_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        checks.expect_runtime_error(
            [&]() { subject.earn(-50); },
            "negative_credit_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.earn(1);
                std::this_thread::sleep_for(5ms);
                subject.redeem(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.points() == 0, "concurrent_transactions");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        checks.expect_runtime_error(
            [&]() { subject.redeem(0); },
            "zero_debit_throws");
    }
    {
        reward_points::reward_account subject{};
        subject.open();
        checks.expect_runtime_error(
            [&]() { subject.earn(0); },
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
