# PoSE

This is the code for the paper PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training

In this work, we introduce **Po**sitional **S**kip-wis**E** (PoSE) training for efficient adaptation of large language models~(LLMs) to extremely long context windows. PoSE decouples train length from target context window size by simulating long inputs using  a fixed context window with manipulated position indices during training.


![PoSE](imgs/pose.png)

Take context window extension from 2,048 to 4,096 as an example, we partition the original context window of 2,048 tokens into two chunks, and adjust the position indices of the second chunk by adding a distinct skipping bias term. These bias terms, as well as the length of each chunk, are altered for each training example, so that the model can adapt to all relative positions of the target context window through fine-tuning.

## What's New

* **[2023/09/19]** Our [paper](paper/PoSE-v1.pdf) and code is released.

## Reproduction
Code and dependencies of our repo can be downloaded as follows:
```
git clone https://github.com/dwzhu-pku/PoSE.git
cd PoSE
pip install -r requirements.txt
```
Additionally, as we utilize [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation on standard benchmarks, please install lm-eval-harness under the `helper/` folder.

### Data, Models and Computation Resources
So far we have experimented with Llama-7B, Llama2-7B and GPT-J-6B. We are currently running experiments to consolidates the effectiveness of PoSE on BaiChuan2.

All the models are fine-tuned on The Pile dataset. Since this dataset is randomly shuffled, we only use the `00` split for training. We further filter the short inputs, and keep 10w samples for fine-tuning, which has proven suffice for our method. 

As for compuatation resources, all our training are conducted on 8 * 32G V100, and all the evaluation is finished on a single A100. 

### Training and Evaluation

scripts under `script/` comprehensively covers the commands for training and evaluation.

For training, the major revision we have made evolves around position indices of the input text. So, you can simply look at the function `train_preprocess_function_pose` to understand our proposed method. There are also some small revisions in `my_modeling_xxx.py` and `my-configuration_xxx.py` for implementing all the linear / NTK / YaRN interpolations and for utilizing xformers for efficient training & inference. Note that we use the revised version of YaRN in our experiments, as supported by the issue [inv_freq seems not calculated right](https://github.com/jquesnelle/yarn/issues/24).

For evaluation, we maded no revision to position indices, so it is completely the same as common setting. 

## Experiment Results
Empirically, we show that PoSE is of great memory and time efficiency:

![efficiency](imgs/efficiency.png)

compatible across various RoPE-based models and interpolation strategies:

![widely_compatible](imgs/widely_compatible.png)

capable of extending to 128k when combined with YaRN interpolation:

![extremely_long](imgs/extremely_long.png)

and receives only minimal performance degradation on standard benchmarks

![standard](imgs/standard.png)

## Citation
