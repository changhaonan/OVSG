# Offical implementation of OVSG

**Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs**, Haonan Chang, Kowndinya Boyalakuntla, Shiyang Lu, Siwei Cai, Eric Jing, Shreesh Keskar, Shijie Geng, Adeeb Abbas, Lifeng Zhou, Kostas Bekris, Abdeslam Boularias

To appear at [CoRL'23](https://www.corl2023.org/).

# Intro

We present an Open-Vocabulary 3D Scene Graph (OVSG), a formal framework for grounding a variety of entities, such as object instances, agents, and regions, with free-form text-based queries. Unlike conventional semantic-based object localization approaches, our system facilitates context-aware entity localization, allowing for queries such as "pick up a cup on a kitchen table" or "navigate to a sofa on which someone is sitting". In contrast to existing research on 3D scene graphs, OVSG supports free-form text input and open-vocabulary querying. Through a series of comparative experiments using the ScanNet dataset and a self-collected dataset, we demonstrate that our proposed approach significantly surpasses the performance of previous semantic-based localization techniques. Moreover, we highlight the practical application of OVSG in real-world robot navigation and manipulation experiments.  The code and dataset used for evaluation will be made available upon publication.

# Example

![OVSG-example](./media/OVSG-L.drawio.png)

# Code-example

## Install Instruction

`OVSG` uses `OVIR-3D` as backbone. So before installing OVSG, you need to install OVIR-3D according to their README file.

After you install `OVIR-3D`, you can install `OVSG` by running:

```
pip install -r requirements.txt
pip install -e .
```

## Running Instruction

First download the demo data from [Link](https://drive.google.com/file/d/1QZH5IuKMuxcTAf4NMJQKzWCr-M26xLer/view?usp=sharing). Put the data into `test_data` folder and extract it to `test_data`. Then we need to run the `OVIR-3D` to perform the fusion first.

### Do OVIR-3D Fusion

Run Detic 2D proposal. For `vocabulary`, we can chooise `lvis`, `ycb_video`, `scannet200`, `imagenet21k`.
```
cd external/OVIR-3D/Detic
python fire.py --dataset ../../../test_data  --vocabulary scannet200
```

Run OVIR-3D fusion
```
cd external/OVIR-3D/src
python fire.py --dataset ../../../test_data --stride 1  --detic_exp scannet200-0.3
```

### Run OVSG

Run OVSG example without LLM:
```
python example/exp_ovsg_llm.py
```

Run OVSG example with LLM:
```
python example/example_ovsg_only.py
```

## Evaluation Code

Being cleaned. Coming soon.

## Trouble Shooting

This repo is being actively maintained, feel free to raise problems in GitHub issues.

# DOVE-G dataset

We also provide a new vision language dataset `DOVE-G`. You can download it from this link: [Link](https://doi.org/10.6084/m9.figshare.24307072.v1).

# Bibtex

If you find `OVSG` to be useful or related with your current work, please cite `OVSG` and `OVIR-3D` by:

```bibtex
@inproceedings{
chang2023contextaware,
title={Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs},
author={Haonan Chang and Kowndinya Boyalakuntla and Shiyang Lu and Siwei Cai and Eric Pu Jing and Shreesh Keskar and Shijie Geng and Adeeb Abbas and Lifeng Zhou and Kostas Bekris and Abdeslam Boularious},
booktitle={7th Annual Conference on Robot Learning},
year={2023},
url={https://openreview.net/forum?id=cjEI5qXoT0}
}
```
```bibtex
@inproceedings{
lu2023ovird,
title={{OVIR}-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data},
author={Shiyang Lu and Haonan Chang and Eric Pu Jing and Abdeslam Boularias and Kostas Bekris},
booktitle={7th Annual Conference on Robot Learning},
year={2023},
url={https://openreview.net/forum?id=gVBvtRqU1_}
}
```