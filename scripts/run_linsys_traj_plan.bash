#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$SCRIPT_DIR/..:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1

python $SCRIPT_DIR/../examples/run_linsys_traj_plan_zero_order.py
python $SCRIPT_DIR/../examples/run_linsys_traj_plan_linear.py
python $SCRIPT_DIR/../examples/run_linsys_traj_plan_cubic.py
