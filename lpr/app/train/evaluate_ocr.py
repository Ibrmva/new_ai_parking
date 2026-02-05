import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, open_dict
import editdistance 
from pytorch_lightning.utilities.model_summary import summarize
from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DEPLOY_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/lpr/ocr_model.pt"

def normalized_edit_distance(pred, target):
    if len(target) == 0:
        return 0 if len(pred) == 0 else 1
    return editdistance.eval(pred, target) / len(target)

def evaluate_dataset(model, dataloader, device):
    model.eval()
    total_ned = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)

        with torch.no_grad():
            logits = model(images)
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            for i in range(images.size(0)):
                pred_label = model.charset_adapter(preds[i])
                true_label = labels[i]

                if pred_label == true_label:
                    correct += 1
                total_ned += normalized_edit_distance(pred_label, true_label)
                total += 1

    accuracy = (correct / total) * 100
    avg_ned = total_ned / total
    return accuracy, avg_ned


@hydra.main(config_path="../../configs", config_name="main", version_base="1.2")
def main(config: DictConfig):
    with open_dict(config):
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)

    model: BaseSystem = hydra.utils.instantiate(config.model)

    if DEPLOY_MODEL_PATH.exists():
        print(f"Loading trained model from {DEPLOY_MODEL_PATH}")
        checkpoint = torch.load(DEPLOY_MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:

            model = checkpoint
    else:
        print("No checkpoint found. Using random weights or pretrained config.")
        if config.get("pretrained") is not None:
            m = model.model if config.model._target_.endswith("PARSeq") else model
            m.load_state_dict(get_pretrained_weights(config.pretrained))

    model.to(DEVICE)
    print("Model summary:")
    print(summarize(model, max_depth=2))

    datamodule = SceneTextDataModule(
        root_dir=config.data.root_dir,
        train_dir=config.data.train_dir,
        val_dir=config.data.val_dir,
        test_dir=config.data.test_dir,
        img_size=config.data.img_size,
        charset_train=config.data.charset_train,
        charset_test=config.data.charset_test,
        max_label_length=config.data.max_label_length,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        augment=False,
        remove_whitespace=config.data.remove_whitespace,
        normalize_unicode=config.data.normalize_unicode,
    )
    datamodule.setup("fit")

    results = {}
    for split_name, loader in [("train", datamodule.train_dataloader()),
                               ("val", datamodule.val_dataloader()),
                               ("test", datamodule.test_dataloader())]:
        if loader is not None:
            acc, ned = evaluate_dataset(model, loader, DEVICE)
            results[split_name] = {"accuracy": acc, "NED": ned}

    evaluated_model_path = DEPLOY_MODEL_PATH.parent / "evaluated_ocr_model.pt"
    print(f"Saving evaluated model to {evaluated_model_path}")
    torch.save(model, evaluated_model_path)

    print("\n===== Evaluation Results =====")
    for split, metrics in results.items():
        print(f"{split.upper()}: Accuracy = {metrics['accuracy']:.2f}%, NED = {metrics['NED']:.4f}")


if __name__ == "__main__":
    main()
