import time
import os
import subprocess

# Every 3 hours a new process is started
max_seconds = 10800
#max_seconds = 900
tstart = time.time()
last_timestep = 0
max_timesteps = 12000000

print("Please pick a model from:")
i = 0
models = [f for f in os.listdir("./osim-rl/osim/models/") if (os.path.isfile(os.path.join("./osim-rl/osim/models/", f)) and f[-5:] == ".osim")]
for model in models:
    print(f"[{i}] : {model}")
    i += 1

print("")
choice = int(input("Please choose your model: "))
exp_model = models[choice]
print(f"You have chosen:\n[{choice}] : {exp_model}")

try:
    os.mkdir(f"./{exp_model[:-5]}")
except FileExistsError as e:
    print("Folder to store data already exists, will overwrite contents of folder.\n")

#Change this if calling the altMain.py and adjust args accordingly
#args = ["mpirun", "-np", "4", "python", "main.py", "0", "0", f"../models/{exp_model}"]
args = ["mpirun", "-np", "2", "python", "altMain.py", "0", "0", f"../models/{exp_model}"]

proc = subprocess.Popen(args)

# Runs for max_timesteps, instead of running for set number of 'games'
while last_timestep < max_timesteps:
    time.sleep(10)
    if time.time() - tstart > max_seconds:
        subprocess.Popen.kill(proc)
        # Get iterations and timesteps from file
        with open(exp_model[:-5] + '/iterations.txt', 'r') as f:
            lines = f.read().splitlines()
            # Get the last line as the last stored iteration
            last_iter = int(lines[-1])
        with open(exp_model[:-5] + '/timesteps.txt', 'r') as g:
            lines = g.read().splitlines()
            # Get the last line as the last stored time step
            last_timestep = int(lines[-1])

        tstart = time.time()
        args = ["mpirun", "-np", "2", "python", "altMain.py", "1", "1", f"../models/{exp_model}"]
        proc = subprocess.Popen(args)

subprocess.Popen.kill(proc)
