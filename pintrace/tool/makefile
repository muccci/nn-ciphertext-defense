# Out-of-tree makefile for the taintblock16trace pintool.
# Layout (relative to this file):
#   ../pin                 Intel Pin kit (64-bit)
#   ./taintblock16trace.cpp
#   ./obj-intel64/taintblock16trace.so   (prebuilt; `make` rebuilds)
#
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PIN_ROOT ?= $(abspath $(THIS_DIR)/../pin)
CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
include $(CONFIG_ROOT)/makefile.config
include makefile.rules
include $(TOOLS_ROOT)/Config/makefile.default.rules
