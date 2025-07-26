#!/bin/bash
export THIS_DIR=$(dirname "$(realpath "$0")")
export PYTHONPATH="$THIS_DIR/openpose_wrapper:$PYTHONPATH"
export LD_LIBRARY_PATH="$THIS_DIR/openpose_wrapper:$LD_LIBRARY_PATH"
python3 pose_joint_estimator_openpose.py