#!/usr/bin/env bash
FILENAME=output/RL/test_before_merge_$(date +"%Y_%m_%d_%I_%M_%p").txt
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")--------------------DEBUG VQAV2------------------------------------------"
src/scripts/sh/debug/debug_script_vqav2.sh   >> $FILENAME 2>&1
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")--------------------DEBUG CLEVR------------------------------------------"
src/scripts/sh/debug/debug_script_clevr.sh   >> $FILENAME 2>&1
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")--------------------DEBUG PRETRAIN------------------------------------------"
src/scripts/sh/debug/debug_script_sl.sh      >> $FILENAME 2>&1
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")--------------------END DEBUG------------------------------------------"
