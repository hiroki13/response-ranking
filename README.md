# Addressee and Response Selection System

This repo contains Theano implementations of the models described in the following papers:

[Addressee and Response Selection for Multi-Party Conversation](https://aclweb.org/anthology/D/D16/D16-1231.pdf). EMNLP 2016

## Addressee and Response Selection Task

#### Dataset
The dataset can be downloaded at [data/input](/data/input).

#### Dependencies
To run the code, you need the following extra packages installed:
  - Numpy and Theano

#### Usage
  1. Clone this repo
  2. Move to the directory [code/](/code): `cd code/`
  3. Run `python -m adr_res_selection.main.main --help` to see all running options

#### Example Comand
  - Static Model: `python -m adr_res_selection.main.main -mode train --train_data ../data/input/train-data.cand-2.gz --dev_data ../data/input/dev-data.cand-2.gz --test_data ../data/input/test-data.cand-2.gz --model static --data_size 100`
  - Dynamic Model: `python -m adr_res_selection.main.main -mode train --train_data ../data/input/train-data.cand-2.gz --dev_data ../data/input/dev-data.cand-2.gz --test_data ../data/input/test-data.cand-2.gz --model dynamic --data_size 100`
