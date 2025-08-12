import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from glob import glob
from dataset import RoadIntersectionDataset
from unet import UNet
from metrics import calculate_metrics
from utils import save_model, load_checkpoint
from losses import BCELoss, FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def avaliar_modelo(model, dataloader, device):
    model.eval()
    ious, precisions, recalls = [], [], []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Avaliando", leave=False)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()

            outputs = torch.sigmoid(model(images))
            preds = outputs > 0.5

            iou, precision, recall = calculate_metrics(preds, masks)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

    return np.mean(ious), np.mean(precisions), np.mean(recalls)

def encontrar_ultimo_checkpoint(pasta_checkpoints):
    arquivos = glob(os.path.join(pasta_checkpoints, "checkpoint_epoch_*.pth"))
    if not arquivos:
        return None
    arquivos.sort(key=os.path.getmtime)
    return arquivos[-1]  # Mais recente

if __name__ == "__main__":
    caminho_train_img = "dataset/train/images"
    caminho_train_mask = "dataset/train/masks"
    caminho_val_img = "dataset/val/images"
    caminho_val_mask = "dataset/val/masks"
    caminho_test_img = "dataset/test/images"
    caminho_test_mask = "dataset/test/masks"
    caminho_checkpoint = "checkpoints"
    epocas = 50
    batch = 4
    salvar_checkpoint_a_cada = 5  # salva checkpoint a cada 5 épocas

    os.makedirs(caminho_checkpoint, exist_ok=True)

    train_dataset = RoadIntersectionDataset(caminho_train_img, caminho_train_mask)
    val_dataset = RoadIntersectionDataset(caminho_val_img, caminho_val_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    bce_loss = BCELoss()
    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_iou = 0.0
    start_epoch = 0

    ultimo_ckpt = encontrar_ultimo_checkpoint(caminho_checkpoint)
    if ultimo_ckpt:
        print(f"🔄 Encontrado checkpoint para retomar: {ultimo_ckpt}")
        checkpoint_data = load_checkpoint(ultimo_ckpt, model, optimizer)
        start_epoch = checkpoint_data['epoch'] + 1
        best_iou = checkpoint_data.get('best_iou', 0.0)
    else:
        print("🚀 Nenhum checkpoint encontrado. Iniciando treinamento do zero.")

    for epoch in range(start_epoch, epocas):
        print(f"\nEpoch {epoch+1}/{epocas}")
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc="Treinando", leave=False)
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).unsqueeze(1).float()

            outputs = model(images)

            loss_bce = bce_loss(outputs, masks)
            loss_focal = focal_loss(outputs, masks)
            loss = loss_bce + loss_focal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Loss médio treino: {train_loss / len(train_loader):.4f}")

        iou_val, prec_val, rec_val = avaliar_modelo(model, val_loader, DEVICE)
        print(f"Val IoU: {iou_val:.4f} | Precisão: {prec_val:.4f} | Recall: {rec_val:.4f}")

        # Salvar checkpoint a cada N épocas
        if (epoch + 1) % salvar_checkpoint_a_cada == 0:
            checkpoint_path = os.path.join(caminho_checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            save_model(model, checkpoint_path, optimizer=optimizer, epoch=epoch, best_iou=best_iou)
            print(f"💾 Checkpoint salvo em {checkpoint_path}")

        # Atualizar o melhor modelo
        if iou_val > best_iou:
            best_iou = iou_val
            best_model_path = os.path.join(caminho_checkpoint, "best_model.pth")
            save_model(model, best_model_path, optimizer=optimizer, epoch=epoch, best_iou=best_iou)
            print(f"🔁 Novo melhor modelo salvo em {best_model_path}")

    print("\n✅ Treinamento finalizado.")

    # Avaliar no conjunto de teste
    if caminho_test_img and caminho_test_mask:
        test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print("\n🔍 Avaliando modelo no conjunto de TESTE...")
        iou_test, prec_test, rec_test = avaliar_modelo(model, test_loader, DEVICE)
        print(f"Test IoU: {iou_test:.4f} | Precisão: {prec_test:.4f} | Recall: {rec_test:.4f}")
