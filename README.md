# Offical implementation of OVSG

**Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs**

**Authors:** [Haonan Chang](https://github.com/changhaonan/), [Kowndinya Boyalakuntla](https://kowndinya2000.github.io), Shiyang Lu, Siwei Cai, Eric Jing, Shreesh Keskar, Shijie Geng, Adeeb Abbas, Lifeng Zhou, Kostas Bekris, [Abdeslam Boularias](http://www.abdeslam.net/)

**Publication Venue:** [CoRL '23](https://www.corl2023.org/)

## Introduction
Welcome to the official implementation of Open-Vocabulary 3D Scene Graphs (OVSG). 

OVSG is a formal framework designed for grounding various entities, including object instances, agents, and regions, using free-form text-based queries. Unlike traditional semantic-based object localization methods, OVSG enables context-aware entity localization, allowing queries like "pick up a cup on a kitchen table" or "navigate to a sofa on which someone is sitting." In contrast to existing 3D scene graph research, OVSG supports open-vocabulary querying and free-form text input. 

We have conducted comparative experiments using the ScanNet dataset and a self-collected dataset [DOVE-G](https://doi.org/10.6084/m9.figshare.24307072.v1), demonstrating that our approach outperforms previous semantic-based localization techniques significantly. Furthermore, we showcase the practical applications of OVSG in real-world robot navigation and manipulation experiments. 

## Illustration of Context-aware Grounding

<p float="left">
  <img src="./media/grounding.png" width="45%" />
  <img src="./media/context_aware_grounding.png" width="45%" /> 
</p>

This is an illustration of the context-aware grounding. Context-aware grounding implies the system should not only consider the object category but also its related entities inside the context. This comparison shows that context-aware grounding can provides a more accurate localization.

## Example

![OVSG Example](./media/OVSG-L.drawio.png)

## Code Example

### Installation Instructions

To get started, clone the OVSG repository:

```bash
git clone https://github.com/changhaonan/OVSG
```
`OVIR-3D` is included as a submodule in `OVSG`. So you'll need to run:

```bash
git submodule update --init --recursive
```

`OVSG` relies on `OVIR-3D` as its backbone, so before installing `OVSG`, please follow the installation instructions provided in the `OVIR-3D` README.

After installing `OVIR-3D`, you can install `OVSG` by running:

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Instructions

Start by downloading the demo data from this [link](https://drive.google.com/file/d/1Gvo-6Lfk93i3NrZ2BahIrUs_BZzIRZyg/view?usp=drive_link). Place the data into the `test_data` folder and extract it there. Then, run `OVIR-3D` to perform the fusion:

### Do OVIR-3D Fusion

Running Detic 2D proposal. For `vocabulary`, we can choose `lvis`, `ycb_video`, `scannet200`, `imagenet21k`.
```bash
cd external/OVIR-3D/Detic
python fire.py --dataset ../../../test_data  --vocabulary scannet200
```

Running OVIR-3D fusion
```bash
cd external/OVIR-3D/src
python fire.py --dataset ../../../test_data --stride 1  --detic_exp scannet200-0.3
```

### Running OVSG

Running `OVSG` example with LLM:

> In the `ovsg/config/api` directory, fill in your OpenAI API key and save the file `openai_demo.yaml` as `openai.yaml` instead and then run:
```bash
python example/exp_ovsg_llm.py
```

Running `OVSG` example without LLM:
```bash
python example/example_ovsg_only.py
```

## Evaluation Code

Checkout [evaluation readme](./evaluation/README.md) for instructions.
Full version is being cleaned and will be released soon.

## Trouble Shooting

This repo is being actively maintained, feel free to raise problems in GitHub issues.

# DOVE-G dataset

We also provide a new vision language dataset `DOVE-G`. You can download it from this link: [Link](https://doi.org/10.6084/m9.figshare.24307072.v1).

## Citation

If `OVSG` and `OVIR-3D` prove useful or relevant to your research, kindly consider citing them using the BibTeX entries below:

```bibtex
@InProceedings{pmlr-v229-chang23b,
  title = 	 {Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs},
  author =       {Chang, Haonan and Boyalakuntla, Kowndinya and Lu, Shiyang and Cai, Siwei and Jing, Eric Pu and Keskar, Shreesh and Geng, Shijie and Abbas, Adeeb and Zhou, Lifeng and Bekris, Kostas and Boularias, Abdeslam},
  booktitle = 	 {Proceedings of The 7th Conference on Robot Learning},
  pages = 	 {1950--1974},
  year = 	 {2023},
  editor = 	 {Tan, Jie and Toussaint, Marc and Darvish, Kourosh},
  volume = 	 {229},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--09 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v229/chang23b/chang23b.pdf},
  url = 	 {https://proceedings.mlr.press/v229/chang23b.html},
  abstract = 	 {We present an Open-Vocabulary 3D Scene Graph (OVSG), a formal framework for grounding a variety of entities, such as object instances, agents, and regions, with free-form text-based queries. Unlike conventional semantic-based object localization approaches, our system facilitates context-aware entity localization, allowing for queries such as “pick up a cup on a kitchen table" or “navigate to a sofa on which someone is sitting". In contrast to existing research on 3D scene graphs, OVSG supports free-form text input and open-vocabulary querying. Through a series of comparative experiments using the ScanNet dataset and a self-collected dataset, we demonstrate that our proposed approach significantly surpasses the performance of previous semantic-based localization techniques. Moreover, we highlight the practical application of OVSG in real-world robot navigation and manipulation experiments. The code and dataset used for evaluation will be made available upon publication.}
}
```

```bibtex
@InProceedings{pmlr-v229-lu23a,
  title = 	 {OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data},
  author =       {Lu, Shiyang and Chang, Haonan and Jing, Eric Pu and Boularias, Abdeslam and Bekris, Kostas},
  booktitle = 	 {Proceedings of The 7th Conference on Robot Learning},
  pages = 	 {1610--1620},
  year = 	 {2023},
  editor = 	 {Tan, Jie and Toussaint, Marc and Darvish, Kourosh},
  volume = 	 {229},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--09 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v229/lu23a/lu23a.pdf},
  url = 	 {https://proceedings.mlr.press/v229/lu23a.html},
  abstract = 	 {This work presents OVIR-3D, a straightforward yet effective method for open-vocabulary 3D object instance retrieval without using any 3D data for training. Given a language query, the proposed method is able to return a ranked set of 3D object instance segments based on the feature similarity of the instance and the text query. This is achieved by a multi-view fusion of text-aligned 2D region proposals into 3D space, where the 2D region proposal network could leverage 2D datasets, which are more accessible and typically larger than 3D datasets. The proposed fusion process is efficient as it can be performed in real-time for most indoor 3D scenes and does not require additional training in 3D space. Experiments on public datasets and a real robot show the effectiveness of the method and its potential for applications in robot navigation and manipulation.}
}
```
