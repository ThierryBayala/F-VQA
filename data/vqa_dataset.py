"""VQA dataset: small subset from HuggingFace (e.g. 5k samples) for pipeline validation."""

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# HuggingFace datasets (pip install datasets)
from datasets import load_dataset  # type: ignore[import-untyped]


def load_vqa_subset(dataset_name: str = "HuggingFaceM4/VQAv2", split: str = "train[:5000]"):
    """Load a small VQA subset from HuggingFace.
    Uses parquet if available (no script); otherwise requires trust_remote_code and datasets<4.0.
    """
    try:
        return load_dataset(
            dataset_name,
            split=split,
            revision="refs/convert/parquet",
        )
    except Exception:
        return load_dataset(dataset_name, split=split, trust_remote_code=True)


class VQADataset(Dataset):
    """Wraps HF VQA subset; returns image, input_ids, attention_mask, labels."""

    def __init__(
        self,
        hf_dataset,
        tokenizer: PreTrainedTokenizerBase,
        image_size: int = 224,
        max_length: int = 256,
        prompt_template: str = "Question: {question} Answer: {answer}",
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get("question", "")
        answer = item.get("multiple_choice_answer", item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else str(item.get("answer", "")))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        text = self.prompt_template.format(question=question, answer=answer)
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        image = item.get("image")
        if image is None:
            from PIL import Image
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        elif not hasattr(image, "size") or image.size[0] == 0:
            from PIL import Image
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
