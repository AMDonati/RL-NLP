#!/usr/bin/env bash
FILENAME=output/RL/test_before_merge_$(date +"%Y_%m_%d_%I_%M_%p").txt
src/scripts/sh/debug/debug_script_vqav2.sh   >> $FILENAME 2>&1
src/scripts/sh/debug/debug_script_clevr.sh   >> $FILENAME 2>&1
src/scripts/sh/debug/debug_script_train.sh   >> $FILENAME 2>&1
