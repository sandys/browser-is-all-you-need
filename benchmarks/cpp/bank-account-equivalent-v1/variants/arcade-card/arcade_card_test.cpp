#include "arcade_card.h"

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
        arcade_card::player_card subject{};
        subject.issue();
        checks.expect(subject.credits() == 0, "newly_started_zero");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(100);
        checks.expect(subject.credits() == 100, "single_credit");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(100);
        subject.load(50);
        checks.expect(subject.credits() == 150, "multiple_credits");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(100);
        subject.spend(75);
        checks.expect(subject.credits() == 25, "single_debit");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(100);
        subject.spend(80);
        subject.spend(20);
        checks.expect(subject.credits() == 0, "multiple_debits");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(100);
        subject.load(110);
        subject.spend(200);
        subject.load(60);
        subject.spend(50);
        checks.expect(subject.credits() == 20, "sequential_operations");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.revoke();
        checks.expect_runtime_error(
            [&]() { (void)subject.credits(); },
            "value_after_stop_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.revoke();
        checks.expect_runtime_error(
            [&]() { subject.load(50); },
            "credit_after_stop_throws");
    }
    {
        arcade_card::player_card subject{};
        checks.expect_runtime_error(
            [&]() { subject.load(50); },
            "credit_before_start_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.revoke();
        checks.expect_runtime_error(
            [&]() { subject.spend(50); },
            "debit_after_stop_throws");
    }
    {
        arcade_card::player_card subject{};
        checks.expect_runtime_error(
            [&]() { subject.revoke(); },
            "stop_before_start_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        checks.expect_runtime_error(
            [&]() { subject.issue(); },
            "start_twice_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(50);
        subject.revoke();
        subject.issue();
        checks.expect(subject.credits() == 0, "restart_resets_zero");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(25);
        checks.expect_runtime_error(
            [&]() { subject.spend(50); },
            "overdraft_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        subject.load(100);
        checks.expect_runtime_error(
            [&]() { subject.spend(-50); },
            "negative_debit_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        checks.expect_runtime_error(
            [&]() { subject.load(-50); },
            "negative_credit_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.load(1);
                std::this_thread::sleep_for(5ms);
                subject.spend(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.credits() == 0, "concurrent_transactions");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        checks.expect_runtime_error(
            [&]() { subject.spend(0); },
            "zero_debit_throws");
    }
    {
        arcade_card::player_card subject{};
        subject.issue();
        checks.expect_runtime_error(
            [&]() { subject.load(0); },
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
