<div align="center">
<h2>Enhancing SeeGround with Relational Depth Text for 3D Visual Grounding</h2>
</div>

- [2025/10]: Manuscript submitted to "Applied Sciences" (MDPI, SCIE-indexed, Impact Factor: 2.5).
- [2025/12]: Manuscript revised 
- [2026/01]: Manuscript accepted
- [2026/01]: This work has been published in the  Applied Sciences

<p align="center">
    <a href='https://doi.org/10.3390/app16020652'><img src="https://img.shields.io/badge/Paper-PDF-blue?style=flat&#x26;logo=doi&#x26;logoColor=yello" alt="Paper PDF"></a>
</p>

🚀 This project base on [SeeGround](https://github.com/iris0329/SeeGround)
---

## Abstract

Three-dimensional visual grounding is a core technology that identifies specific objects within complex 3D scenes based on natural language instructions, enhancing human–machine interactions in robotics and augmented reality domains. Traditional approaches have focused on supervised learning, which relies on annotated data; however, zero-shot methodologies are emerging due to the high costs of data construction and limitations in generalization. SeeGround achieves state-of-the-art performance by integrating 2D rendered images and spatial text descriptions. Nevertheless, SeeGround exhibits vulnerabilities in clearly discerning relative depth relationships owing to its implicit depth representations in 2D views. This study proposes the relational depth text (RDT) technique to overcome these limitations, utilizing a Monocular Depth Estimation model to extract depth maps from rendered 2D images and applying the K-Nearest Neighbors algorithm to convert inter-object relative depth relations into natural language descriptions, thereby incorporating them into Vision–Language Model (VLM) prompts. This method distinguishes itself by augmenting spatial reasoning capabilities while preserving SeeGround’s existing pipeline, demonstrating a 3.54% improvement in the Acc@0.25 metric on the Nr3D dataset in a 7B VLM environment that is approximately 10.3 times lighter than the original model, along with a 6.74% increase in Unique cases on the ScanRefer dataset, albeit with a 1.70% decline in Multiple cases. The proposed technique enhances the robustness of grounding through viewpoint anchoring and candidate discrimination in complex query scenarios, and is expected to improve efficiency in practical applications through future multi-view fusion and conditional execution optimizations.


## Keywords
3D Visual Grounding, Vision-Language Models, Zero-Shot Grounding, Open-Vocabulary Learning, Spatial Reasoning, Monocular Depth Estimation, Relational Depth Text, Depth-Aware Grounding

<img width="5100" height="3871" alt="image" src="https://github.com/user-attachments/assets/37837392-0007-4181-bbaf-153f4ced3f94" />


### Table 1. Computational cost comparison between the baseline and our method. The proposed method requires only marginal additional resources (approx. +40ms per query latency). Meas-urements were conducted on a single NVIDIA GeForce RTX 3090.
| **Metric** | **Dataset** | **Reproduced 7B** | **Ours 7B+Depth** | **Overhead (Diff)** |
| :--- | :---: | :---: | :---: | :---: |
| VRAM Usage (GB) | - | 20.16 | 22.56 | + 2.40 |
| Total Inference Time | Nr3D | 5h 04m | 5h 10m | + 6m (+1.9%) |
| Total Inference Time | ScanRefer | 20m | 21m | + 1m (+5.0%) |

### Table 2. Performance comparison on the Nr3D benchmark (Unit: %). Easy/Hard categorizes per-formance by query difficulty (number of distractors), while Dep./Indep. by viewpoint dependency. Ours 7B+Depth represents the performance of the proposed methodology with the Relational Depth Text module, while Reproduced 7B is the lightweight baseline reproduced in our hardware environment. Ori Baseline values are cited from the original SeeGround paper, with 72B serving as an upper-bound reference, and '-' indicating unavailable values.
| **Method** | **Easy** | **Hard** | **Dep.** | **Indep.** | **Acc@25** | **Acc@50** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ori Baseline 72B | 54.50 | 38.30 | 42.30 | 48.20 | 46.10 | - |
| Ori Baseline 7B | 40.80 | 26.30 | 31.40 | 34.30 | 33.30 | - |
| Reproduced 7B | 40.99 | 25.97 | 31.35 | 34.17 | 33.18 | 32.88 |
| **Ours 7B+Depth** | **44.39** | **29.65** | **34.59** | **37.87** | **36.72** | **36.34** |

### Table 3. Performance comparison on the ScanRefer benchmark (Unit: %). Unique/Multiple cate-gorizes performance by the presence of identical class objects in the scene. The rest of the con-figuration is the same as in Table 2.
| **Method** | **Unique@25** | **Multiple@25** | **Unique@50** | **Multiple@50** | **Acc@25** | **Acc@50** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ori Baseline 72B | 75.70 | 34.00 | 68.90 | 30.00 | 44.10 | 39.40 |
| Ori Baseline 7B | - | - | - | - | - | - |
| Reproduced 7B | 64.61 | **28.46** | 60.67 | **26.38** | 37.59 | **35.04** |
| **Ours 7B+Depth** | **71.35** | 26.76 | **65.17** | 23.91 | **38.01** | 34.33 ||

### Table 4. Comparison with state-of-the-art zero-shot methods on Nr3D and ScanRefer benchmarks. (Unit: %). 'SeeGround (Baseline)' refers to the results reproduced in our environment using the 7B model. Due to space constraints, the 'Type' column is omitted; however, all compared methods operate in a zero-shot manner. Note that for the Nr3D dataset, most SOTA methods report only the Overall Accuracy (identification rate), so the IoU-based Acc@0.50 metric is not applicable (-). '†' indicates results cited from the original papers.
| **Method** | **Backbone (Size)** | **Nr3D (Acc@25)** | **ScanRefer (Acc@25)** | **ScanRefer (Acc@50**) |
| :--- | :--- | :---: | :---: | :---: |
| VLM-Grounder | GPT-4V (Large) | 48.00 † | 51.60 † | 32.80 † |
| SORT3D | GPT-4o (Large) | 62.00 † | - | - |
| SeqVLM | Doubao-1.5-vision-pro | 53.20 † | 55.60 † | 49.60 † |
| View-on-Graph | Qwen2-VL-72B | 47.60 † | 44.80 † | 40.30 † |
| SeeGround (Baseline) | Qwen2-VL-7B | 33.18 | 37.59 | 35.04 |
| **Ours 7B+Depth** | **Qwen2-VL-7B** | **36.72** | **38.01** | **34.33** |
