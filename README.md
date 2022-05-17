# pytorch-fpn-segmentation


### train model
```
python train.py --num_epoch=[EPOCH] --root_dir=[PATH] --load_checkpoint[MODEL_PATH]
```

### evaluation
```
python eval.py --model_path=[MODEL_PATH]
```

### tensorboard
```
%reload_ext tensorboard

%tensorboard --logdir [DIR]
```
