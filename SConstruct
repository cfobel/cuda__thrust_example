import os

env = Environment()
env.Tool('cuda')
env.Append(CPPPATH=os.environ['CPATH'].split(':'))
env.Program(['thrust_mapped.cu'], LIBS=['cuda', 'cudart'])
