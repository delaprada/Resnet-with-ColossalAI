# Train ResNet on CIFAR-10 from scratch
This example provides a training script and an evaluation script. The training script provides an example of training ResNet on CIFAR10 dataset from scratch.

### Possible bugs discovered in the colossalai package
For `ColossalAI/colossalai/initialize.py` line 55:
```
init_method = f"tcp://[{host}]:{port}"
```
The original code has brackets on '{host}', which will cause errors when executing the colossai command. Removing the brackets can solve this problem.

Also I suggest having additional tutorial for the environment path setting. For example, adding this line of code to your `~/.bashrc`:
```
export PATH="$HOME/.local/bin:$PATH"
```

 Since without setting the environment path, the 'colossalai: command not found' error will alert.

### Used model and dataset

The used model is resnet18 and dataset is CIFAR10.

### Reproduction results

| Model     | Booster DDP with FP32 | Booster DDP with FP16 | Booster Low Level Zero | Booster Gemini |
| --------- | --------------------- | --------------------- | ---------------------- | -------------- |
| ResNet-18 | 85.22%                | 85.16%                | 84.63%                 | 84.63%         |

For complete training and evaluation logs, please refer to: `train_log_fp32.txt`, `train_log_fp16.txt`, `train_log_low_level_zero.txt` and `test_log_fp16.txt`, `test_log_fp32.txt`


### Install requirements

```bash
pip install -r requirements.txt
```

### Train
```bash
# train with torch DDP with fp32
colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp16 -p torch_ddp_fp16

# train with low level zero
colossalai run --nproc_per_node 1 train.py -c ./ckpt-low_level_zero -p low_level_zero

# train with gemini
colossalai run --nproc_per_node 1 train.py -c ./ckpt-gemini -p gemini
```

### Eval

```bash
# evaluate fp32 training
python eval.py -c ./ckpt-fp32 -e 80

# evaluate fp16 mixed precision training
python eval.py -c ./ckpt-fp16 -e 80

# evaluate low level zero training
python eval.py -c ./ckpt-low_level_zero -e 80

# evaluate gemini training
python eval.py -c ./ckpt-gemini -e 80
```

Expected accuracy performance will be:

| Model     | Single-GPU Baseline FP32 | Booster DDP with FP32 | Booster DDP with FP16 | Booster Low Level Zero | Booster Gemini |
| --------- | ------------------------ | --------------------- | --------------------- | ---------------------- | -------------- |
| ResNet-18 | 85.85%                   | 84.91%                | 85.46%                | 84.50%                 | 84.60%         |
