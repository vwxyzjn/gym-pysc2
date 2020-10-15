# gym-pysc2

Gym wrapper for pysc2

## Installation

```
# First step: install SC2 
## type `iagreetotheeula` to unzip SC2
wget -O ~/sc2.zip http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && \
unzip ~/sc2.zip -d ~/ && \
rm ~/sc2.zip
# hack: fix the error message
# ffmpeg: ~/StarCraftII/Libs/libstdc++.so.6: version `GLIBCXX_3.4.26'
# but make sure you have ffmpeg installed
mv ~/StarCraftII/Libs/libstdc++.so.6 ~/StarCraftII/Libs/libstdc++.so.6.temp

# Second step: install pysc2 
git clone https://github.com/vwxyzjn/gym-pysc2
cd gym-pysc2
pip install -e .
```

## Get started

To test installation, run the following code.

```
python ppo_sc2.py \
    --gym-id SC2MoveToBeacon-v0 \
    --num-envs 1 \
    --num-steps 256 \
    --wandb-project-name cleanrl.benchmark \
    --wandb-entity cleanrl --cuda True \
    --capture-video
```

Then open another terminal to check tensorboard

```
tensorboard --logdir runs
```

You could also check the videos of agents playing the game
in the `videos` folder


# Benchmark results

This is still a work in progress but you can see some
example runs here

https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Gym-pysc2-Benchmark--VmlldzoyNTEyMTc