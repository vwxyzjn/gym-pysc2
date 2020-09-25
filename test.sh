python ppo_sc2.py \
    --gym-id SC2MoveToBeacon-v0 \
    --num-envs 8 \
    --num-steps 256 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode \
    --wandb-entity cleanrl --cuda True \
    --capture-video

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_sc2.py \
    --gym-id SC2CollectMineralShards-v0 \
    --num-envs 8 \
    --num-steps 256 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed) >& /dev/null &
done
