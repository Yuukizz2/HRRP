import torch
import numpy as np
from torch.utils.data import DataLoader
from model.hrrp_model import HRRPModel
from model.atten_classify import AttentionClassifier
from dataset.hrrp_dataset import HrrpDataset
from data.embedding_tensor import build_label_embeddings_tensor
from train_zeroshot import test


def test_zeroshot(ckpt_path, embedding_dir, test_data_dir, label_jsons, aspect_names, batch_size, num_workers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_labels = checkpoint['train_labels']
    test_labels = checkpoint['test_labels']
    label_to_aircraft = checkpoint.get('label_to_aircraft', None)
    label_embeddings_tensor = build_label_embeddings_tensor(embedding_dir, label_jsons, aspect_names, label_to_aircraft)
    classifier = AttentionClassifier(
        hrrp_dim=512,
        num_labels=8,
        label_aspects=11,
        label_embedding_dim=768,
        label_embeddings_tensor=label_embeddings_tensor,
        trainable_labels=True
    ).to(device)
    hrrp_model = HRRPModel(hrrp_len=512, feature_dim=512).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    hrrp_model.load_state_dict(checkpoint['hrrp_model_state_dict'])
    test_dataset = HrrpDataset(
        data_path=test_data_dir,
        return_format="dict"
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_acc, seen_acc, unseen_acc = test(
        classifier, hrrp_model, test_loader, device, train_labels, test_labels
    )
    print(f"总准确率: {total_acc:.4f}")
    print(f"Seen类别准确率: {seen_acc:.4f}")
    print(f"Unseen类别准确率: {unseen_acc:.4f}")


if __name__ == "__main__":
    # ========== 配置参数 ==========
    ckpt_path = "ckpt/zeroshot_20250811_150226/best_model.pth"  # 你的模型权重路径
    embedding_dir = "data/embedding/Qwen_QwQ-32B"
    test_data_dir = "data/test"
    label_jsons = [
        "A321_hierarchical_embedding.json",
        "A330_hierarchical_embedding.json",
        "BY787_hierarchical_embedding.json",
        "BY737_hierarchical_embedding.json",
        "Z8_hierarchical_embedding.json",
        "ARJ21_hierarchical_embedding.json",
        "KJ600_hierarchical_embedding.json",
        "Y20_hierarchical_embedding.json"
    ]
    aspect_names = [
        "flightCharacteristics", "microDopplerSignature", "basicInformation",
        "thermalSignature", "roleAndMission", "geometricCharacteristics",
        "materialsAndCoatings", "aircraftModel", "radarCrossSection",
        "engineAndPerformance", "global"
    ]
    batch_size = 8
    num_workers = 8

    test_zeroshot(ckpt_path, embedding_dir, test_data_dir, label_jsons, aspect_names, batch_size, num_workers)
