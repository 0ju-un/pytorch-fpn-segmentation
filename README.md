# pytorch-fpn-segmentation

### tutorial

[velog](https://velog.io/@0ju-un/PyTorch%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-FPN%EC%9C%BC%EB%A1%9C-%ED%95%9C%EA%B8%80-%EC%86%90%EA%B8%80%EC%94%A8%EC%97%90%EC%84%9C-%EC%9E%90%EB%AA%A8-%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0-dataset-%EC%A0%9C%EC%9E%91%EB%B6%80%ED%84%B0-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5-%EC%98%88%EC%B8%A1%EA%B9%8C%EC%A7%80)


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
