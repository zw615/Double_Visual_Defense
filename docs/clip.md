## **CLIP model usage**
Our clip repo is built on [OpenCLIP](https://github.com/mlfoundations/open_clip).
The original $\Delta$CLIP models were trained on TPU using JAX and their weights were converted into PyTorch-compatible format for user convenience.
We upload the weight to (huggingface)[TBD] and provide the following example code snippet with OpenCLIP:

```python
import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained('hf-hub:zw123/delta_clip_h14_336')
tokenizer = get_tokenizer('hf-hub:zw123/delta_clip_h14_336')

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)

text = tokenizer(["a diagram", "a dog", "a cat", "a beignet"], context_length=model.context_length)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0., 0., 0., 1.0]]
```

We have also provided some example scripts under path `CLIP_benchmark/scripts` for both zero-shot clean performance and robustness evaluation.