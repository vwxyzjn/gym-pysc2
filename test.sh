python ppo_sc2.py \
    --num-envs 8 \
    --num-steps 256 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode \
    --wandb-entity cleanrl --cuda True \
    --capture-video