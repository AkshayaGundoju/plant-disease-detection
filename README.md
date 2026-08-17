\# Plant Disease Detection — CNN Image Classifier



Identifies tomato leaf diseases from a photograph using transfer learning on

MobileNetV2, with Grad-CAM explanations.



\*\*Live demo:\*\* https://plant-disease-detection-uhlau7uh69b7qrdaud6rrs.streamlit.app



\## Results



| Model | Test accuracy | Macro-F1 |

|---|---|---|

| Baseline CNN (from scratch) | 0.821 | — |

| MobileNetV2 — frozen base | 0.875 | — |

| MobileNetV2 — fine-tuned | \*\*0.945\*\* | \*\*0.927\*\* |



10 tomato classes, \~18,000 images, 80/10/10 split.



Worst-performing class: \*\*Early Blight\*\* (F1 0.849), mainly confused with

Target Spot and Septoria Leaf Spot — three diseases that all present as

similar brown/dark leaf lesions.



\## Approach



1\. `tf.data` pipeline with prefetching (caching disabled on the training

&#x20;  set to avoid RAM overflow on free-tier hardware)

2\. Augmentation — flips, rotation, zoom, brightness/contrast jitter

3\. Baseline CNN from scratch to establish a floor (82.1% val accuracy)

4\. Transfer learning — MobileNetV2 (ImageNet), frozen base, new head,

&#x20;  lr=1e-3 (87.5% val accuracy)

5\. Fine-tuning — top 40 layers unfrozen, BatchNorm kept frozen, lr=1e-5

&#x20;  (94.5% test accuracy)

6\. Grad-CAM to verify the model attends to lesions, not background



\## Why MobileNetV2



The deployment target is a phone in a field. MobileNetV2 uses depthwise

separable convolutions (\~8-9x cheaper than standard convolution) and is

\~14 MB, versus \~98 MB for ResNet50 — which also matters for free-tier

hosting.



\## Honest limitations



\- \*\*PlantVillage is a laboratory dataset\*\*: single detached leaves on

&#x20; uniform backgrounds. Published work shows models trained on it degrade

&#x20; sharply on real field photographs. The reported accuracy should NOT be

&#x20; read as field accuracy. (In informal testing on a real field photo, the

&#x20; model still correctly identified Early Blight at 97% confidence — a

&#x20; good sign, but not a substitute for proper field validation.)

\- Only tomato (10 classes); the pipeline generalises to all 38.

\- No "not a leaf" class — any image is forced into a known class.

&#x20; Mitigated with a confidence threshold warning in the UI.



\## Run locally



```bash

python -m venv venv

source venv/bin/activate   # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

streamlit run app.py

```



\## Tech stack



Python, TensorFlow/Keras, MobileNetV2, Streamlit, Google Colab (free GPU

for training)

