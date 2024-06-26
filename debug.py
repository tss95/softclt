
import os

# Define the command and parameters
bs=8
data='CharacterTrajectories'
tau_inst=10
tau_temp=2.0

command = f"python softclt_ts2vec/train.py {data} --loader='UEA' --batch-size {bs} --eval --tau_inst {tau_inst} --tau_temp {tau_temp}"

# Run the command
os.system(command)