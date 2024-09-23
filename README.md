These two codes are modified version based on repositories from https://github.com/oxwhirl/pymarl and https://github.com/azure-123/Backdoor-FACMAC, with additional references to the code from https://github.com/Hyperparticle/one-pixel-attack-keras.

# Run an experiment 
## Train models based on clean observations

```shell
cd Discrete action space
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z Number_attack=0
```

```shell
cd Continuous action space
python src/main.py --config=facmac_mamujoco --env-config=mujoco_multi with env_args.scenario_name="Ant-v2" env_args.agent_conf="4x2" env_args.agent_obsk=1 Number_attack=0
```

## SJAV Training

```shell
cd Discrete action space
python src/main.py --config=q_adv --env-config=sc2 with env_args.map_name=2s3z Number_attack=0 attack_method=q_adv_de
```

```shell
cd Continuous action space
python src/main.py --config=q_adv_mamujoco --env-config=mujoco_multi with env_args.scenario_name="Ant-v2" env_args.agent_conf="4x2" env_args.agent_obsk=1 Number_attack=0 attack_method=q_adv_de
```

## Run RTCA attack and AMCA attack

```shell
cd Discrete action space
python src/main.py --config=q_adv --env-config=sc2 with env_args.map_name=2s3z Number_attack=1 evaluate=True test_nepisode=100 test_nepisode=500 Number_attack=1 attack_method=q_adv_de attack_target_method=fgsm checkpoint_path_q_adv=xxx checkpoint_path=xxx
```

```shell
cd Continuous action space
python src/main.py --config=q_adv_mamujoco --env-config=mujoco_multi with env_args.scenario_name="Ant-v2" env_args.agent_conf="4x2" env_args.agent_obsk=1 attack_method=q_adv_de evaluate=True test_nepisode=500 Number_attack=1 attack_method=q_adv_de attack_target_method=sgld checkpoint_path_q_adv=xxx checkpoint_path=xxx
```

```tex
@inproceedings{ecai_ZhouL23_RTCA,
  author       = {Ziyuan Zhou and
                  Guanjun Liu},
  title        = {Robustness Testing for Multi-Agent Reinforcement Learning: State Perturbations
                  on Critical Agents},
  booktitle    = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence,
                  September 30 - October 4, 2023, Krak{\'{o}}w, Poland - Including
                  12th Conference on Prestigious Applications of Intelligent Systems
                  {(PAIS} 2023)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {372},
  pages        = {3131--3139},
  publisher    = {{IOS} Press},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230632},
  doi          = {10.3233/FAIA230632}
}
```
、、、
@ARTICLE{TSMCA_AMCA,
  author={Zhou, Ziyuan and Liu, Guanjun and Guo, Weiran and Zhou, MengChu},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Adversarial Attacks on Multiagent Deep Reinforcement Learning Models in Continuous Action Space}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TSMC.2024.3454118}}
、、、


