# Semantic Audio-Visual Embodied Navigation (saven) Model

## Usage

### SAVEN 
- Pre-training the vision model:
```
python ss_baselines/saven/pretraining/vision_model_trainer.py --run-type train
```

- Pre-training the audio model:
```
python ss_baselines/saven/pretraining/audio_model_trainer.py --run-type train
```

- Pre-train the `saven` model (using the pre-trained vision and audio model). `Saven` is first trained with the external memory size of 1, which only uses the last observation:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven
```

- Evaluate the pre-training process. This will automatically run evaluation on the `val_seen-scenes_heard-sounds` data split for each of the checkpoints found in `data/models/saven/data`:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven --run-type eval EVAL.SPLIT val_seen-scenes_heard-sounds
```
Use the additional flag `--prev-ckpt-ind` to instead specify a starting checkpoint index `n` for the evaluation process, or to resume an evaluation process:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --prev-ckpt-ind n --model-dir data/models/saven --run-type eval EVAL.SPLIT val_seen-scenes_heard-sounds
```

- Once evaluation is complete, obtain the best checkpoint of the pre-training step and its corresponding metrics:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven --run-type eval --eval-best EVAL.SPLIT val_seen-scenes_heard-sounds
```

- Train the `saven` model using the best pre-trained checkpoint of pre-training it. Please update the `pretrained_weights` path in [saven.yaml](config/semantic_audionav/saven.yaml) with the best pre-trained checkpoint when finetuning:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml --model-dir data/models/saven
```

### Run the Seq2Seq baselines
This code includes the configuration files, policy and trainer to run six different Seq2Seq baselines:
- Train PointGoal (RGB + Depth + GPS):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/pointgoal/pointgoal.yaml --model-dir data/models/pointgoal
```
- Train ObjectGoal (RGB + Depth + GPS + Semantic Label):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/objectgoal/objectgoal.yaml --model-dir data/models/objectgoal
```
- Train AudioPointGoal (Audio + RGB + Depth + GPS):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-pointgoal/audio-pointgoal.yaml --model-dir data/models/audio-pointgoal
```
- Train AudioObjectGoal (Audio + RGB + Depth + GPS + Semantic Label):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-objectgoal/audio-objectgoal.yaml --model-dir data/models/audio-objectgoal
```
- Train AudioGoal (Audio + RGB + Depth):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-pointgoal/audiogoal.yaml --model-dir data/models/audiogoal
```
- Train AudioObjectGoal-NoGPS (Audio + RGB + Depth + Semantic Label):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-objectgoal/audio-objectgoal_no-gps.yaml --model-dir data/models/audio-objectgoal_no-gps
```
In our paper we report the results obtained using `AudioGoal` and `AudioObjectGoal-NoGPS`.

To evaluate either of the baselines add the flag ```--run-type eval``` and specify the appropriate evaluation split, e.g., ```EVAL.SPLIT val_seen-scenes_heard-sounds```. 

### Run the random baselines
There are two random baselines `RandomAgentWithoutStop` and `RandomAgentWithStop`. The former is random baseline that uniformly samples one of three actions (FORWARD, LEFT, RIGHT) and executes stop when the radius distance is less than the specified success distance (1 meter): 
```
python ss_baselines/saven/run.py --run-type eval --exp-config ss_baselines/saven/config/random_agent_wo-stop.yaml
```
The latter samples one of four actions (FORWARD, LEFT, RIGHT, STOP) where STOP has a much lower probability of being selected:
```
python ss_baselines/saven/run.py --run-type eval --exp-config ss_baselines/saven/config/random_agent_w-stop.yaml
```
In our paper we report the results obtained using `RandomAgentWithoutStop`.

## Notes 
- Modify the parameter `NUM_UPDATES` in the configuration file according to the number of GPUs
