export ENV_DEMO02_ROOT=/home/wangyitong/code/demo02
mpirun -n 1 $ENV_DEMO02_ROOT/build/test_slu ./files/AA1.mtx
mpirun -n 1 $ENV_DEMO02_ROOT/build/test_pangulu ./files/AA1.mtx
mpirun -n 1 $ENV_DEMO02_ROOT/build/test_cudss ./files/AA1.mtx
mpirun -n 1 $ENV_DEMO02_ROOT/build/test_strumpack ./files/AA1.mtx
