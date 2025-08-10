# 02_treinar_unet.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import RoadIntersectionDataset
from unet import UNet
from metrics import calculate_metrics
from utils import save_model
from losses import BCELoss, FocalLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def avaliar_modelo(model, dataloader, device, writer=None, epoch=None):
    model.eval()
    ious, precisions, recalls = [], [], []
    sample_images, sample_preds, sample_masks = None, None, None

    with torch.no_grad():
        loop = tqdm(dataloader, desc='Avaliando', leave=False)
        for idx, (images, masks) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            outputs = torch.sigmoid(model(images))
            preds = outputs > 0.5

            iou, precision, recall = calculate_metrics(preds, masks)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

            # Guardar um batch para logar no TensorBoard
            if idx == 0:
                sample_images = images.cpu()
                sample_preds = preds.float().cpu()
                sample_masks = masks.cpu()

    mean_iou = np.mean(ious)
    mean_prec = np.mean(precisions)
    mean_rec = np.mean(recalls)

    if writer and epoch is not None:
        writer.add_scalar('Val/IoU', mean_iou, epoch)
        writer.add_scalar('Val/Precision', mean_prec, epoch)
        writer.add_scalar('Val/Recall', mean_rec, epoch)

        # Logar imagens
        writer.add_images('Val/Images', sample_images, epoch)
        writer.add_images('Val/Predictions', sample_preds, epoch)
        writer.add_images('Val/Masks', sample_masks, epoch)

    return mean_iou, mean_prec, mean_rec


if __name__ == '__main__':
    caminho_train_img = 'dataset/train/images'
    caminho_train_mask = 'dataset/train/masks'
    caminho_val_img = 'dataset/val/images'
    caminho_val_mask = 'dataset/val/masks'
    caminho_test_img = 'dataset/test/images'
    caminho_test_mask = 'dataset/test/masks'
    caminho_checkpoint = 'checkpoints'

    epocas = 6
    batch = 6
    salvar_checkpoint_a_cada = 5
    checkpoint_para_retomar = None  # "checkpoints/checkpoint_epoch_5.pth"

    os.makedirs(caminho_checkpoint, exist_ok=True)
    writer = SummaryWriter(log_dir='runs/treinamento_unet')

    # Datasets e loaders
    train_dataset = RoadIntersectionDataset(caminho_train_img, caminho_train_mask)
    val_dataset = RoadIntersectionDataset(caminho_val_img, caminho_val_mask)
    test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Modelo, perdas e otimizador
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    bce_loss = BCELoss()
    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    best_iou = 0.0

    # Retomar de checkpoint, se existir
    if checkpoint_para_retomar and os.path.exists(checkpoint_para_retomar):
        checkpoint = torch.load(checkpoint_para_retomar, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"🔄 Retomando treino do epoch {start_epoch} (melhor IoU até agora: {best_iou:.4f})")

    # Loop de treino
    for epoch in range(start_epoch, epocas):
        print(f"\nEpoch {epoch+1}/{epocas}")
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc='Treinando', leave=False)
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            outputs = model(images)
            loss_bce = bce_loss(outputs, masks)
            loss_focal = focal_loss(outputs, masks)
            loss = loss_bce + loss_focal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_loader)
        print(f"Loss médio treino: {mean_train_loss:.4f}")
        writer.add_scalar('Train/Loss', mean_train_loss, epoch)

        # Histograma dos pesos
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Avaliação
        iou_val, prec_val, rec_val = avaliar_modelo(model, val_loader, DEVICE, writer, epoch)
        print(f"Val IoU: {iou_val:.4f} | Precisão: {prec_val:.4f} | Recall: {rec_val:.4f}")

        # Checkpoint periódico
        if (epoch + 1) % salvar_checkpoint_a_cada == 0:
            checkpoint_path = os.path.join(caminho_checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_train_loss,
                'best_iou': best_iou
            }, checkpoint_path)
            print(f"💾 Checkpoint salvo em {checkpoint_path}")

        # Melhor modelo
        if iou_val > best_iou:
            best_iou = iou_val
            best_model_path = os.path.join(caminho_checkpoint, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_train_loss,
                'best_iou': best_iou
            }, best_model_path)
            print(f"🔁 Novo melhor modelo salvo em {best_model_path}")

    print('\n✅ Treinamento finalizado.')

    # Avaliação no teste
    print('\n🔍 Avaliando modelo no conjunto de TESTE...')
    iou_test, prec_test, rec_test = avaliar_modelo(model, test_loader, DEVICE, writer, epocas)
    print(f"Test IoU: {iou_test:.4f} | Precisão: {prec_test:.4f} | Recall: {rec_test:.4f}")

    writer.close()
