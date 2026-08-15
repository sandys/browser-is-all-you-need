#include "print_spool.h"

#include "catch.hpp"

TEST_CASE("a fresh spool is idle") {
    spool::PrintSpool spooler(3);
    REQUIRE(spooler.idle());
    REQUIRE_FALSE(spooler.full());
    REQUIRE(spooler.backlog() == 0);
    REQUIRE(spooler.slots() == 3);
}

TEST_CASE("zero slots is rejected") {
    REQUIRE_THROWS_AS(spool::PrintSpool(0), std::invalid_argument);
}

TEST_CASE("releasing an idle spool throws") {
    spool::PrintSpool spooler(2);
    REQUIRE_THROWS_AS(spooler.release(), spool::SpoolEmpty);
}

TEST_CASE("queueing a full spool throws") {
    spool::PrintSpool spooler(2);
    spooler.queue("alpha");
    spooler.queue("beta");
    REQUIRE(spooler.full());
    REQUIRE_THROWS_AS(spooler.queue("gamma"), spool::SpoolFull);
}

TEST_CASE("jobs are released in arrival order") {
    spool::PrintSpool spooler(3);
    spooler.queue("alpha");
    spooler.queue("beta");
    spooler.queue("gamma");
    REQUIRE(spooler.release() == "alpha");
    REQUIRE(spooler.release() == "beta");
    REQUIRE(spooler.release() == "gamma");
    REQUIRE(spooler.idle());
}

TEST_CASE("a full spool is neither idle nor queueable but is releasable") {
    spool::PrintSpool spooler(1);
    spooler.queue("solo");
    REQUIRE(spooler.full());
    REQUIRE_FALSE(spooler.idle());
    REQUIRE(spooler.release() == "solo");
    REQUIRE(spooler.idle());
    REQUIRE_FALSE(spooler.full());
}

TEST_CASE("displace drops the longest waiting job when full") {
    spool::PrintSpool spooler(2);
    spooler.queue("alpha");
    spooler.queue("beta");
    spooler.displace("gamma");
    REQUIRE(spooler.backlog() == 2);
    REQUIRE(spooler.release() == "beta");
    REQUIRE(spooler.release() == "gamma");
}

TEST_CASE("displace behaves like queue when there is room") {
    spool::PrintSpool spooler(3);
    spooler.queue("alpha");
    spooler.displace("beta");
    REQUIRE(spooler.backlog() == 2);
    REQUIRE(spooler.release() == "alpha");
    REQUIRE(spooler.release() == "beta");
}

TEST_CASE("slots are reused without corruption") {
    spool::PrintSpool spooler(3);
    for (int cycle = 0; cycle < 40; ++cycle) {
        const std::string job = "job-" + std::to_string(cycle);
        spooler.queue(job);
        REQUIRE(spooler.release() == job);
    }
    REQUIRE(spooler.idle());
}

TEST_CASE("backlog tracks queue and release") {
    spool::PrintSpool spooler(4);
    spooler.queue("alpha");
    spooler.queue("beta");
    REQUIRE(spooler.backlog() == 2);
    spooler.release();
    REQUIRE(spooler.backlog() == 1);
}

TEST_CASE("abandon clears a partially filled spool") {
    spool::PrintSpool spooler(3);
    spooler.queue("alpha");
    spooler.queue("beta");
    spooler.abandon();
    REQUIRE(spooler.idle());
    REQUIRE(spooler.backlog() == 0);
    REQUIRE_THROWS_AS(spooler.release(), spool::SpoolEmpty);
}
