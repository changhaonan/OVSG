# Evaluation of OVSG

**Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs**

**Authors:** [Haonan Chang](https://github.com/changhaonan/), [Kowndinya Boyalakuntla](https://kowndinya2000.github.io), Shiyang Lu, Siwei Cai, Eric Jing, Shreesh Keskar, Shijie Geng, Adeeb Abbas, Lifeng Zhou, Kostas Bekris, [Abdeslam Boularias](http://www.abdeslam.net/)

**Publication Venue:** [CoRL '23](https://www.corl2023.org/)

## Overview
This README provides a detailed guide on evaluating Open-Vocabulary 3D Scene Graphs (OVSG). By adhering to the following instructions, you can replicate the results mentioned in the paper.

## Evaluation Guide

### Prerequisites
Ensure the necessary data from ScanNet and DOVE-G is appropriately placed in the evaluation directory structure:
- ScanNet scenes should be placed in `eval_data/scannet`
- [DOVE-G](https://doi.org/10.6084/m9.figshare.24307072.v1) scenes should be placed in `eval_data/dove-g`

### Evaluation on ScanNet
For example, if you've downloaded the ScanNet scene `scene0011_00`, execute the following commands in the terminal:

1. **Generate Scene Graph and Query Match Data:**
   ```bash
   python evaluation/eval_scannet.py app=env_only env=notionovidb_scannet task_name=\[\'gen_query:scene0011_00\'\] exp_name=exp_scene0011_00
   ```
   
2. **Evaluate System Post Scene Graph Generation:**
   ```bash
   python evaluation/eval_scannet.py app=env_only env=notionovidb_scannet task_name=\[\'eval_query:scene0011_00\'\] exp_name=exp_scene0011_00
   ```
   
3. **(Alternate Idea) Execute Both Steps in One Command:**
   ```bash
   python evaluation/eval_scannet.py app=env_only env=notionovidb_scannet task_name=\[\'gen_n_eval_query:scene0011_00\'\] exp_name=exp_scene0011_00
   ```

### Evaluation on DOVE-G
Analogously, for the DOVE-G scene `kitchen`:

1. **Generate Scene Graph and Ground-Truth Matchings:**
   ```bash
   python evaluation/eval_scannet.py app=env_only env=notionovidb_dove-g task_name=\[\'gen_query:kitchen\'\] exp_name=exp_kitchen
   ```
   
2. **Evaluate System Post Scene Graph Generation:**
   ```bash
   python evaluation/eval_scannet.py app=env_only env=notionovidb_dove-g task_name=\[\'eval_query:kitchen\'\] exp_name=exp_kitchen
   ```
   
3. **(Alternate Idea) Execute Both Steps in One Command:**
   ```bash
   python evaluation/eval_scannet.py app=env_only env=notionovidb_dove-g task_name=\[\'gen_n_eval_query:kitchen\'\] exp_name=exp_kitchen
   ```
   

## Citation

If `OVSG` and `OVIR-3D` prove useful or relevant to your research, kindly consider citing them using the BibTeX entries below:


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