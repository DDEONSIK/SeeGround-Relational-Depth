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

3D visual grounding is a core technology that identifies specific objects within complex 3D scenes based on natural language instructions, enhancing human-machine interactions in robotics and augmented reality domains. Traditional approaches have focused on supervised learning reliant on annotated data; however, due to data construction costs and generalization limitations, zero-shot methodologies are emerging, with SeeGround achieving state-of-the-art performance by integrating 2D rendered images and spatial text descriptions. Nevertheless, SeeGround exhibits vulnerabilities in clearly discerning relative depth relationships owing to its implicit depth representations in 2D views. This study proposes the Relational Depth Text technique to overcome these limitations, utilizing a Monocular Depth Estimation model to extract depth maps from rendered 2D images and applying the K-Nearest Neighbors algorithm to convert inter-object relative depth relations into natural language descriptions, thereby incorporating them into VLM prompts. This method distinguishes itself by augmenting spatial reasoning capabilities while preserving SeeGround's existing pipeline, demonstrating a 3.54% improvement in the Acc@0.25 metric on the Nr3D dataset in a 7B VLM environment that is approximately 10.3 times lighter than the original model, along with a 6.74% increase in Unique cases on the ScanRefer dataset, albeit with a 1.70% decline in Multiple cases. The proposed technique enhances the robustness of grounding through viewpoint anchoring and candidate discrimination in complex query scenarios, and is anticipated to expand efficiency in practical applications via future multi-view fusion and conditional execution optimizations.


## Keywords
3D Visual Grounding, Vision-Language Models, Zero-Shot Grounding, Open-Vocabulary Learning, Spatial Reasoning, Monocular Depth Estimation, Relational Depth Text, Depth-Aware Grounding

<img width="5821" height="4419" alt="image" src="https://github.com/user-attachments/assets/24d89326-e0c8-407b-97ac-be8464cd45bc" />


### Table 1. Performance comparison on the Nu3D benchmark (Unit: %). Easy/Hard categorizes performance by query difficulty (number of distractors), while Dep./Indep. by viewpoint dependency. Ours HW 7B+Depth represents the performance of the proposed methodology with the Relational Depth Text module, while Ours HW 7B is the lightweight baseline reproduced in our hardware environment. Ori Baseline values are cited from the original SeoGround paper, with 72B serving as an upper-bound reference, and '-' indicating unavailable values.
| Method | Easy | Hard | Dep. | Indep. | Acc@25 | Acc@50 |
|--------|-----------|-------------|-----------|-------------|--------|--------|
| Ori Baseline 72B | 54.50 | 38.30 | 42.30 | 48.20 | 46.10 | - |
| Ori Baseline 7B | 40.80 | 26.30 | 31.40 | 34.30 | 33.30 | - |
| Ours HW 7B | 40.99 | 25.97 | 31.35 | 34.17 | 33.18 | 32.88 |
| Ours HW 7B+Depth | 44.39 | 29.65 | 34.59 | 37.87 | 36.72 | 36.34 |

### Table 2. Performance comparison on the ScanRefer benchmark (Unit: %). Unique/Multiple categorizes performance by the presence of identical class objects in the scene. The rest of the configuration is the same as in Table 1.
| Method | Unique@25 | Multiple@25 | Unique@50 | Multiple@50 | Acc@25 | Acc@50 |
|--------|-----------|-------------|-----------|-------------|--------|--------|
| Ori Baseline 72B | 75.70 | 34.00 | 68.90 | 30.00 | 44.10 | 39.40 |
| Ori Baseline 7B | - | - | - | - | - | - |
| Ours HW 7B | 64.61 | 28.46 | 60.67 | 26.38 | 37.59 | 35.04 |
| Ours HW 7B+Depth | 71.35 | 26.76 | 65.17 | 23.91 | 38.01 | 34.33 |
