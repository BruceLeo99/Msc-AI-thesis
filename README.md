## Multimodal Prototype-Based Networks for Interpretable Food Classification (Food-101)

> Novel, multimodal ProtoPNet-style PBN using VisualBERT + generated captions (BLIP/BLIP2) to improve Food-101 classification accuracy while preserving interpretability.

### Highlights
- **Novelty**: A new, multimodal Prototype-Based Network (mPBN) that fuses images with generated captions, extending **ProtoPNet** [1] with a **VisualBERT** [2] encoder. Prototypes live in the joint semantic space of text–vision.
- **Interpretability**: Classify-by-prototypes with explicit cluster and separation terms; additional **DeepSHAP** [3][4] analyses for both unimodal and multimodal models.
- **Results**: On Food-101, the best multimodal model (BLIP2 captions) achieves **77.34%** accuracy, surpassing the best unimodal ProtoPNet (**69.74%**) and VGG16 baseline (**55.37%**).
- **Engineering**: End-to-end pipeline: caption generation (BLIP/BLIP2), data curation, efficient dataloading, multimodal fusion, 3-stage ProtoPNet training, evaluation, and explainability artifacts.

<p style="text-align:center"><b>Original PBN Architecture</p>
<figure>
  <img width="800" height="600" alt="PBN Architecture" 
       src="https://github.com/user-attachments/assets/ab64c2f4-9ce5-4534-b4a6-96faabc4c643"
       style="display:block; margin:auto;" />
</figure>

<br><br><br>

<p style="text-align:center"><b>Novel mPBN Architecture</p>
<figure>
  <img width="800" height="600" alt="mPBN Architecture" 
       src="https://github.com/user-attachments/assets/f57c0dd4-8ccc-40a3-8f79-4b7866dcfb04"
       style="display:block; margin:auto;" />
</figure>


---

## 1) Overview
- **Objective**: Build an interpretable, high-accuracy food classifier that leverages both image content and natural-language descriptions, enabling prototype-level reasoning grounded in multimodal semantics.
- **Key idea**: Generate captions for each image (BLIP/BLIP2) [5][6], tokenize them, and feed the text together with CNN features into **VisualBERT**. Learn class-specific prototypes in the VisualBERT [CLS] space and classify by proximity to these prototypes.
- **Findings**:
  - Using captions substantially improves Food-101 accuracy vs image-only baselines.
  - Caption quality matters: BLIP2 captions outperform BLIP.
  - A small number of prototypes per class (1–2) works best in the multimodal setting; too many prototypes can hurt generalization.

For more context, see the final thesis submission `MSc_AI_Thesis_yyg760.pdf`.

---

## 2) Methodology

### Data & Captions
- **Dataset**: Food-101 (101 classes) [7]. Custom annotations and label mappings are used.
  - Dataloader: `Food101_dataloader.py` with `Food101Dataset` and helpers `lazy_load_original` / `lazy_load_customized`.
  - Inputs: images normalized to VGG/ResNet stats; labels mapped via `Food101/food-101/meta/label_map.json`.
- **Captioning**: `Food101_captioning.py` with pretrained Hugging Face models (BLIP and BLIP-2)
  - Captions are cleaned (noise removal + label leakage prevention) and saved into JSON annotations in `complete_annotations/`.

### Models 

#### Multimodal PBN (mPBN)
- Core: `multimodal_PBN.py`
  - Visual features: `VGG16` or "ResNet34" backbone features projected to match VisualBERT’s expected dimension.
  - Text features: tokenized captions (`bert-base-uncased`) into **VisualBERT** (`uclanlp/visualbert-vqa-coco-pre`).
  - Fusion: Visual token (from VGG16) + text tokens in VisualBERT → use `[CLS]` embedding as joint representation.
  - Prototypes: learn class-specific prototypes in the `[CLS]` space; logits are negative distances to prototypes via a linear last layer.

#### Baselines & Comparisons
- Image-only baselines: `vgg16_model.py`, `resnet34.py`.
- Unimodal ProtoPNet (image-only): `unimodal_ProtoPNet.py` (also three-stage training, spatial prototypes).

#### Training flow for baseline PBN and mPBN: 
- `ProtoPNet/train_and_test_mPBN.py` with the canonical ProtoPNet three stages
  1. Warm-up (train add-on/prototypes/last layer)
  2. Joint training (unfreeze all: encoder, CNN features, projection, prototypes, last layer)
  3. Last-layer optimization (optional convex-like step, L1 on last layer)
- Losses: Cross-entropy + prototype cluster/separation terms (+ optional L1), following ProtoPNet design (`ProtoPNet/settings.py`).
- Current project trained with 1P, 2P, 5P and 10P (_x_P = _x_ prototypes per class, see `MSc_AI_Thesis_yyg760.pdf` for more detail).

### Explainability: DeepSHAP for both unimodal and multimodal
  - Unimodal: `DeepSHAP_explaner_PBN*.py`.
  - Multimodal: `DeepSHAP_explaner_mPBN*.py`,  (wrappers prepare visual + repeated text inputs).
---

## 3) Technical Configuration & Installation

### Main libraries 
- Python, PyTorch (e.g., torch 2.7.0 CUDA), torchvision, scikit-learn, numpy, pandas, matplotlib, scikit-image
- Hugging Face Transformers ([VisualBERT](https://github.com/uclanlp/visualbert), [BLIP](https://huggingface.co/docs/transformers/en/model_doc/blip), [BLIP2](https://huggingface.co/docs/transformers/en/model_doc/blip-2))
- [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet)
  
See full list in `requirements.txt`

### Repository Map (selected)
- Multimodal model: `multimodal_PBN.py`
- Unimodal ProtoPNet and baselines: `unimodal_ProtoPNet.py`, `vgg16_model.py`, `resnet34.py`
- ProtoPNet core: `ProtoPNet/` (settings, train/test stages, push/prune, etc.)
- Data loading: `Food101_dataloader.py`
- Captioning: `Food101_captioning.py`
- SHAP explainers: `DeepSHAP_explaner_*`
- Execution scripts (to train models on servers, setup model configs, and execute DeepSHAP analysis. .py scripts and .sh files):
  - Model training: `model_training_scripts/*`
  - Model training: `SHAP_scripts/*`
- Results: `results_final/` (per-experiment folders, CSV/JSON reports)

### Data Preparation
- Download Food-101 and organize metadata as in `Food101/food-101/meta/`([Original download link](https://www.kaggle.com/datasets/dansbecker/food-101). 
- Generate captions using `Food101_captioning.py` (BLIP or BLIP2) and merge into your annotation JSONs as `blip_caption` / `blip2_caption`.
- Confirm label mappings in `Food101/food-101/meta/label_map.json`.

### Training
- Multimodal training entry points:
  - Programmatic: call `train_multimodal_PBN(...)` from `multimodal_PBN.py`.
  - The training stages and optimizers are orchestrated by `ProtoPNet/train_and_test_mPBN.py` with hyperparameters in `ProtoPNet/settings.py`.
- Unimodal ProtoPNet: `train_protopnet(...)` in `unimodal_ProtoPNet.py`.
- Baselines: `train_vgg16(...)`,  `train_resnet34(...)`.
- Evaluation: each module provides `test_*` functions and checkpoints (.pth file); aggregated CSV/JSON are written into `results_final/`(not published, please contact the author for complete result).
- All trainings were completed on DAS-6 Supercomputer [8].
  
Example (pseudo-code):
```python
from Food101_dataloader import lazy_load_original
from multimodal_PBN import train_multimodal_PBN, test_multimodal_PBN

train, test = lazy_load_original(load_captions=True, transform='vgg16', target_transform='integer', caption_type='blip2')
model_path = train_multimodal_PBN(train, test, model_name='mPBN_1p_blip2', device='cuda', num_prototypes_per_class=1, num_epochs=50, save_result=True)
results = test_multimodal_PBN(model_path, 'mPBN_1p_blip2', test, device='cuda', save_result=True)
print(results) # Or log the result
```

---

## 4) Results & Conclusion

### Key Results (Food-101)

| Model                            | Accuracy (%) | Notes                  |
|----------------------------------|--------------|------------------------|
| Baseline VGG16                   | 55.37        | Image-only             |
| PBN-ResNet34 (2 prototypes)      | 69.74        | Best unimodal PBN      |
| mPBN (BLIP captions, 1 prototype)| 72.65        | Multimodal (BLIP)      |
| mPBN (BLIP2 captions, 1 prototype)| 77.34       | Best overall (BLIP2)   |

(To access the complete result, please check `MSc_AI_Thesis_yyg760.pdf` or contact the author)

### Findings
- **Multimodality helps**: Adding captions lifts accuracy by ~7–8 points over the best unimodal ProtoPNet.
- **Caption quality matters**: BLIP2 > BLIP, consistent with stronger language modeling.
- **Prototype count**: 1–2 per class works best for mPBN. Larger counts can overfit and degrade performance.
- **Interpretability**: Prototypes in the joint space remain human-inspectable; SHAP analyses further explain visual/text contributions per prediction.

### Conclusion
- This project demonstrates that aligning prototypes with multimodal semantics produces a more accurate and still interpretable classifier. Language grounding via high-quality generated captions is a practical and scalable path to boost performance without human annotations. However, the performance of PBN models with multimodality can vary across several factors, such as the quality of the caption, the number of prototypes, model complexity, etc. Future study is required to discover a "gonden standard" to improve robustness of such multimodality task.

---

## Getting Started

1) Create environment and install deps
** Python version: 3.12.7**
```bash
pip install -r requirements.txt
```
2) Prepare Food-101 and annotation JSONs, including generated captions.
3) Train a model via the script in  `model_training_scripts/*` (see example above).
4) Visualize the model's attention in  `SHAP_scripts/*`

Hardware: a CUDA-enabled GPU is strongly recommended for BLIP2/VisualBERT and training.

---

## Explainability Artifacts
- Run SHAP explainers to visualize pixel- and token-level influences:
  - Multimodal: `DeepSHAP_explaner_mPBN*.py`
  - Unimodal: `DeepSHAP_explaner_PBN*.py`
- Classification reports and confusion matrices are auto-saved in `results_final/`.

---

## References & Acknowledgements
<ol>
<li>Chen, C., Li, O., Tao, A., Barnett, A., Su, J., Rudin, C.: This looks like that:
 Deep learning for interpretable image recognition. arXiv preprint arXiv:1806.10574
 (2018), https://arxiv.org/abs/1806.10574</li>
<li>Li, L. H., Yatskar, M., Yin, D., Hsieh, C., & Chang, K. (2019). VisualBERT: A Simple and Performant Baseline for Vision and Language. ArXiv. https://arxiv.org/abs/1908.03557</li>
<li> Liu, M., Ning, Y., Yuan, H., Ong, M.E.H., Liu, N.: Balanced background and expla
nation data are needed in explaining deep learning models with shap: An empirical
 study on clinical decision making (2022), https://arxiv.org/abs/2206.04050</li>
<li> Lundberg, S., Lee, S.I.: A unified approach to interpreting model predictions
(2017), https://arxiv.org/abs/1705.07874</li>
<li>Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre
training for unified vision-language understanding and generation. In: Proceed
ings of the 39th International Conference on  Machine Learning (ICML) (2022),
 https://arxiv.org/abs/2201.12086</li>
<li> Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pre
training with frozen image encoders and large language models. In: Krause, A.,
 Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceed
ings of the 40th International Conference on Machine Learning. Proceedings of
 Machine Learning Research, vol. 202, pp. 19730–19742. PMLR (23–29 Jul 2023),
 https://proceedings.mlr.press/v202/li23q.html</li>
<li>Bossard, L., Guillaumin, M., Gool, L.V.: Food-101– mining discriminative compo
nents with random forests. In: Proceedings of the European Conference on Com
puter Vision (ECCV). pp. 446–461. Springer, Zurich, Switzerland (2014)</li>
<li>Henri Bal, Dick Epema, Cees de Laat, Rob van Nieuwpoort, John Romein, Frank
 Seinstra, Cees Snoek, and Harry Wijshoff: "A Medium-Scale Distributed System for
 Computer Science Research: Infrastructure for the Long Term", IEEE Computer,
 Vol. 49, No. 5, pp. 54-63, May 2016.
</li>
</ol>

For research and implementation details, see the thesis PDF: `MSc_AI_Thesis_yyg760.pdf`.

---

## Notes for Reviewers (Engineering Readiness)
- Clear separation of concerns: data prep, captioning, dataloading, training, evaluation, and explainability.
- ProtoPNet patterns preserved for reproducibility; multimodal extension is modular and configurable.
- Results are scripted with CSV/JSON artifacts for easy comparison and auditability. 
