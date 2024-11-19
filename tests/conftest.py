# This file enables the ipydex excepthook together with pytest.
# The custom excepthook can be activated by `activate_ips_on_exception()`
# in your test-file.

# To prevent unwanted dependencies the custom excepthook is only active if a
# special environment variable is "True". Use the following command for this:
#
# export PYTEST_IPS=True


import os
import pytest
import argparse

if os.getenv("PYTEST_IPS") == "True":
    import ipydex

    # This function is just an optional reminder
    def pytest_runtest_setup(item):
        print("This invocation of pytest is customized")

    def pytest_exception_interact(node, call, report):
        ipydex.ips_excepthook(call.excinfo.type, call.excinfo.value, call.excinfo.tb, leave_ut=True)


# register custom marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "optional_dependency: mark test to require argument --opt-dep to run"
    )


def pytest_addoption(parser):
    parser.addoption("--opt-dep", action="store_true", help="Run tests for optional dependencies", default=False)


def pytest_collection_modifyitems(config, items):
    opt_dep_flag = config.getoption("--opt-dep") or False

    # Skip some tests if --all is not provided
    reason_true = "Skipping {}. Do not use --opt-dep to run."
    reason_false = "Skipping {}. Use --opt-dep to run."

    for item in items:
        if opt_dep_flag:
            reason = reason_true.format(item.nodeid)
        else:
            reason = reason_false.format(item.nodeid)
        marker = pytest.mark.skip(reason=reason)

        condition = ("optional_dependency" not in item.keywords)

        if condition == opt_dep_flag:
            item.add_marker(marker)
            # print("marked: ", item.nodeid)
