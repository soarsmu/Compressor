# Compressor

This replication package contains the source code for fine-tuning pre-trained models, model simplification and training, as well as all trained compressed models.

## Environment configuration

We provide a Dockerfile to help build the experimental environment. Please run the following scripts to to compile a docker image:
```
docker build -t YOUR_CUSTOM_TAG .
```
Be careful with the torch version that you need to use, modify the Dockerfile according to your cuda version pls.

Then, please run the docker:
```
dokcer run -it -v YOUR_LOCAL_REPO_PATH:/root/Compressor --gpus all YOUR_CUSTOM_TAG
```

After that, pls go inside the docker first, and then install some necessary libraries:

```
pip3 install -r requirements.txt
```

GraphCodeBERT need a parser to extract data flows from the source code, please go to `./parser` to compile the parser first. Pls run:
```
cd parser
bash build.sh
```

## How to run

The scripts and instructions for each experiment are in the `README.md` files under each subfolder.

## Misc

If you meet any problems when using the tool, please contact Jieke SHI by [jiekeshi@smu.edu.sg](mailto:jiekeshi@smu.edu.sg). Many thanks!

If you use any code from our repo, pls cite:

```
@misc{https://doi.org/10.48550/arxiv.2208.07120,
  url = {https://arxiv.org/abs/2208.07120},
  author = {Shi, Jieke and Yang, Zhou and Xu, Bowen and Kang, Hong Jin and Lo, David},
  title = {Compressing Pre-trained Models of Code into 3 MB},
  publisher = {arXiv},
  year = {2022}
}

```
