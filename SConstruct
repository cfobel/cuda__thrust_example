import os

env = Environment()
env.Tool('cuda')
env.Append(CPPPATH=os.environ['CPATH'].split(':'))
env.Program(['cuda_info.cpp', 'CudaInfo.cpp'], LIBS=['cuda', 'cudart'])
