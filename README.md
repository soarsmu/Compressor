# Compressor

This replication package contains the source code for fine-tuning pre-trained models, model simplification and training, as well as all trained compressed models.

## Environment configuration

Please use a docker container with `PyTorch version >= 1.6`. For example,

```
docker run -it --gpus all -v <your repo path>:/workspace/Compressor --name <your container name> pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
```

Which version of PyTorch should be used may be related to the version of cuda on your machine, please check carefully. Note that you can set `--cpus=<num>` to limit the used cpus in testing inference latency.

Then, please install some necessary libraries:

```
pip install -r requirements.txt
```

## Directory structure

```
.
├── data
│   ├── clond_detection (processed data)
│   └── vulnerability_prediction (sampled and processed data)
│ 
├── src
│   ├── clond_detection
│   │   ├── baseline (trained models of the baseline)
│   │   ├── checkpoint (trained models of compressed models)
│   │   ├── compress (code for compressing models, including searching architecture and knowledge distillation)
│   │   └── finetune (code for finetuning CodeBERT on clone detection)
│   │
│   └── vulnerability_prediction
│       ├── baseline (trained models of the baseline)
│       ├── checkpoint (trained models of compressed models)
│       ├── compress (code for compressing models, including searching architecture and knowledge distillation)
│       └── finetune (code for finetuning CodeBERT on vulnerability prediction)
│ 
├── src_gcb
│   ├── clond_detection
│   │   ├── baseline (trained models of the baseline)
│   │   ├── checkpoint (trained models of compressed models)
│   │   └── finetune (code for compressing and finetuning GraphCodeBERT on clone detection)
│   │
│   └── vulnerability_prediction
│       ├── baseline (trained models of the baseline)
│       ├── checkpoint (trained models of compressed models)
│       └── finetune (code for compressing and finetuning GraphCodeBERT on clone detection)
│ 
└──python_parser (a parser to turn code into data flow graph, specifically designed for GraphCodeBERT, from its offical repository)
```

## How to run

The scripts for each experiments are in the `<src or src_gcb>/<task>/README.md`.
