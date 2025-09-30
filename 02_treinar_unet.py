# 02_treinar_unet.py
# ---------------------------------------------------------------------------------------
# 💡 TensorBoard:
#    1) python 02_treinar_unet.py
#    2) tensorboard --logdir runs --port 6006
#    3) http://localhost:6006
# ---------------------------------------------------------------------------------------

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from glob import glob

from components.dataset import RoadIntersectionDataset
from components.unet import UNet
from components.metrics import calculate_metrics
from components.utils import save_model, load_checkpoint
from components.losses import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BANDS_MODE = "rgbnir"      # "rgbnir" (padrão) ou "rgb"
ARCH = "smp_unet"            # "custom" (sua UNet) ou "smp_unet"

def buscar_melhor_threshold(model, val_loader, device, thresholds=np.linspace(0.1, 0.9, 17)):
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    preds_list, targs_list = [], []
    with torch.inference_mode():
        for imgs, masks in tqdm(val_loader, desc="Coletando predições", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            probs = sigmoid(model(imgs))
            preds_list.append(probs)
            targs_list.append(masks)
    all_probs = torch.cat(preds_list, dim=0)
    all_targs = torch.cat(targs_list, dim=0)

    best_t, best_iou = 0.5, -1.0
    for t in thresholds:
        bin_preds = (all_probs > t).to(dtype=torch.uint8)
        iou, _, _ = calculate_metrics(bin_preds, all_targs)
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return float(best_t), float(best_iou)

def avaliar_modelo(model, dataloader, device, threshold=0.5):
    model.eval()
    ious, precisions, recalls = [], [], []
    with torch.inference_mode():
        for images, masks in tqdm(dataloader, desc="Avaliando", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).to(dtype=torch.uint8)
            iou, precision, recall = calculate_metrics(preds, masks)
            ious.append(iou); precisions.append(precision); recalls.append(recall)
    return float(np.mean(ious)), float(np.mean(precisions)), float(np.mean(recalls))

def encontrar_ultimo_checkpoint(pasta_checkpoints):
    arquivos = glob(os.path.join(pasta_checkpoints, "checkpoint_epoch_*.pth"))
    if not arquivos:
        return None
    arquivos.sort(key=os.path.getmtime)
    return arquivos[-1]

def build_model(arch: str, in_ch: int):
    if arch == "custom":
        return UNet(in_channels=in_ch, out_channels=1).to(DEVICE)
    elif arch == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
        except Exception as e:
            raise RuntimeError("Para usar ARCH='smp_unet', instale segmentation_models_pytorch.") from e
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_ch, classes=1).to(DEVICE)
    else:
        raise ValueError(f"ARCH inválido: {arch}")

def adapt_first_conv_if_needed(model, checkpoint_state):
    """
    Ajusta a 1ª conv quando o checkpoint foi treinado com 3 canais e o modelo atual tem 4 (ou vice-versa).
    """
    state = checkpoint_state.get('model_state_dict', checkpoint_state)
    model_state = model.state_dict()
    possible_keys = [k for k in model_state.keys() if k.endswith("weight") and model_state[k].dim() == 4]
    if not possible_keys:
        model.load_state_dict(state, strict=False)
        return
    first_conv_key = possible_keys[0]
    if first_conv_key not in state:
        model.load_state_dict(state, strict=False)
        return
    w_ckpt = state[first_conv_key]
    w_model = model_state[first_conv_key]
    in_ckpt, in_model = w_ckpt.shape[1], w_model.shape[1]
    if in_ckpt == in_model:
        model.load_state_dict(state, strict=False)
        return
    if in_ckpt == 3 and in_model == 4:
        w_new = w_model.clone()
        w_new[:, :3, :, :] = w_ckpt
        w_new[:, 3:4, :, :] = w_ckpt.mean(dim=1, keepdim=True)
        state[first_conv_key] = w_new
    elif in_ckpt == 4 and in_model == 3:
        state[first_conv_key] = w_ckpt[:, :3, :, :].contiguous()
    model.load_state_dict(state, strict=False)

if __name__ == "__main__":
    caminho_train_img = "dataset_mini/train/images"
    caminho_train_mask = "dataset_mini/train/masks"

    caminho_val_img = "dataset_mini/val/images"
    caminho_val_mask = "dataset_mini/val/masks"

    caminho_test_img = "dataset_mini/test/images"
    caminho_test_mask = "dataset_mini/test/masks"

    caminho_checkpoint = "checkpoints"
    epocas = 1000
    batch = 64
    lr = 1e-4
    weight_decay = 1e-2
    salvar_checkpoint_a_cada = 50

    swa_start_epoch = 50
    swa_lr = 1e-5

    patience_val = 30
    patience_train = 10
    min_delta = 1e-6

    os.makedirs(caminho_checkpoint, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train_dataset = RoadIntersectionDataset(
        caminho_train_img, caminho_train_mask, is_training=True, bands_mode=BANDS_MODE
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True,
        pin_memory=True, prefetch_factor=4, num_workers=8
    )

    val_loader = None
    if os.path.exists(caminho_val_img) and os.path.exists(caminho_val_mask):
        val_dataset = RoadIntersectionDataset(
            caminho_val_img, caminho_val_mask, is_training=False, bands_mode=BANDS_MODE
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            pin_memory=True, prefetch_factor=4, num_workers=8
        )

    test_loader = None
    if os.path.exists(caminho_test_img) and os.path.exists(caminho_test_mask):
        test_dataset = RoadIntersectionDataset(
            caminho_test_img, caminho_test_mask, is_training=False, bands_mode=BANDS_MODE
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    in_ch = 3 if BANDS_MODE == "rgb" else 4
    model = build_model(ARCH, in_ch)

    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epocas if len(train_loader) > 0 else epocas
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    best_iou = 0.0
    best_threshold = 0.5
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    start_epoch = 0

    ultimo_ckpt = encontrar_ultimo_checkpoint(caminho_checkpoint)
    if ultimo_ckpt:
        print(f"🔄 Encontrado checkpoint para retomar: {ultimo_ckpt}")
        checkpoint_data = load_checkpoint(ultimo_ckpt, model, optimizer)
        adapt_first_conv_if_needed(model, checkpoint_data)
        start_epoch = checkpoint_data.get('epoch', -1) + 1
        best_iou = checkpoint_data.get('best_iou', 0.0)
        best_threshold = checkpoint_data.get('best_threshold', 0.5)
        best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
    else:
        print("🚀 Nenhum checkpoint encontrado. Iniciando treinamento do zero.")

    val_patience_counter = 0
    train_patience_counter = 0
    using_train_fallback = False

    writer = SummaryWriter(log_dir=f"runs/Deteccao_Cruzamento_Rodovias_{BANDS_MODE}_{ARCH}")
    writer.add_text(
        "Hiperparametros",
        f"epochs={epocas}, batch_size={batch}, lr={lr}, weight_decay={weight_decay}, "
        f"loss=FocalLoss(0.8,2), model={ARCH}, in_ch={in_ch}, optimizer=AdamW, "
        f"scheduler=OneCycleLR, SWA_start={swa_start_epoch}, SWA_lr={swa_lr}, "
        f"early_stopping=LossVal_then_LossTrain(patience_val={patience_val}, patience_train={patience_train}), "
        f"bands_mode={BANDS_MODE}, min_delta={min_delta}"
    )

    stopped_epoch = None
    global_step = start_epoch * max(1, len(train_loader))

    USE_AMP = True
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    for epoch in range(start_epoch, epocas):
        print(f"\nEpoch {epoch+1}/{epocas}")
        model.train()
        train_loss_sum = 0.0

        loop = tqdm(train_loader, desc="Treinando", leave=False)
        for step, (images, masks) in enumerate(loop):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True).float()

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
                loss = focal_loss(outputs, masks)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch < swa_start_epoch:
                scheduler.step()

            train_loss_sum += float(loss.item())
            loop.set_postfix(loss=float(loss.item()), lr=optimizer.param_groups[0]['lr'])

            writer.add_scalar("Loss/treino_step", float(loss.item()), global_step)
            global_step += 1

        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        current_lr = float(optimizer.param_groups[0]['lr'])
        print(f"Loss médio treino: {avg_train_loss:.4f} | LR: {current_lr:.2e}")

        writer.add_scalar("Loss/treino", float(avg_train_loss), epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=USE_AMP):
                for images, masks in val_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True).float()
                    outputs = model(images)
                    val_loss_sum += float(focal_loss(outputs, masks).item())
            val_loss = val_loss_sum / max(1, len(val_loader))
            print(f"Loss médio validação: {val_loss:.4f}")
            writer.add_scalar("Loss/validacao", float(val_loss), epoch)

            best_t_epoca, _ = buscar_melhor_threshold(model, val_loader, DEVICE)
            iou_val, prec_val, rec_val = avaliar_modelo(model, val_loader, DEVICE, threshold=best_t_epoca)
            print(f"Melhor threshold: {best_t_epoca:.2f} | IoU: {iou_val:.4f} | Precisão: {prec_val:.4f} | Recall: {rec_val:.4f}")
            writer.add_scalar("Metricas/val_iou", float(iou_val), epoch)
            writer.add_scalar("Metricas/val_precision", float(prec_val), epoch)
            writer.add_scalar("Metricas/val_recall", float(rec_val), epoch)
            writer.add_scalar("Metricas/val_best_threshold", float(best_t_epoca), epoch)
            writer.add_scalars("Metricas/validacao", {
                "IoU": float(iou_val),
                "Precisao": float(prec_val),
                "Recall": float(rec_val)
            }, epoch)

            improved_val = val_loss < (best_val_loss - min_delta)
            if improved_val:
                best_val_loss = float(val_loss)
                best_threshold = float(best_t_epoca)
                val_patience_counter = 0
                using_train_fallback = False

                best_model_path = os.path.join(caminho_checkpoint, "best_model.pth")
                save_model(
                    model, best_model_path, optimizer=optimizer, epoch=epoch,
                    best_iou=float(iou_val),
                    best_threshold=best_threshold,
                    best_val_loss=best_val_loss
                )
                print(f"🔁 Novo melhor modelo salvo! ValLoss: {best_val_loss:.4f} | IoU(val): {iou_val:.4f}")
                writer.add_scalar("Metricas/best_val_loss_so_far", best_val_loss, epoch)
            else:
                val_patience_counter += 1
                print(f"⚠️ ValLoss não melhorou ({val_loss:.4f} >= {best_val_loss:.4f}) | patience_val {val_patience_counter}/{patience_val}")

                if val_patience_counter >= patience_val and not using_train_fallback:
                    using_train_fallback = True
                    train_patience_counter = 0
                    print("➡️  Alternando para early stopping por Loss de Treino (fallback).")

            if (epoch + 1) % salvar_checkpoint_a_cada == 0:
                checkpoint_path = os.path.join(caminho_checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
                save_model(
                    model, checkpoint_path, optimizer=optimizer, epoch=epoch,
                    best_iou=float(iou_val), best_threshold=float(best_t_epoca), best_val_loss=best_val_loss
                )
                print(f"💾 Checkpoint salvo em {checkpoint_path}")

        if using_train_fallback or (val_loader is None):
            improved_train = avg_train_loss < (best_train_loss - min_delta)
            if improved_train:
                best_train_loss = float(avg_train_loss)
                train_patience_counter = 0
                writer.add_scalar("Metricas/best_train_loss_so_far", best_train_loss, epoch)
                print(f"✅ Treino melhorou: best_train_loss = {best_train_loss:.4f}")
            else:
                train_patience_counter += 1
                print(f"⏳ Treino não melhorou | patience_train {train_patience_counter}/{patience_train}")

                if train_patience_counter >= patience_train:
                    print("🛑 Parando treino: Loss de Treino parou de melhorar no fallback.")
                    stopped_epoch = epoch + 1
                    writer.add_scalar("EarlyStopping/Stopped_epoch", float(stopped_epoch), epoch)
                    writer.flush()
                    break

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"🔄 SWA atualizado (época {epoch+1})")

        writer.flush()

    print("\n✅ Treinamento finalizado.")
    if stopped_epoch is not None:
        print(f"📌 Early stopping ativado na época {stopped_epoch}.")

    if (val_loader is not None) and (start_epoch < swa_start_epoch < epocas):
        print("\n🔄 Finalizando SWA...")
        update_bn(val_loader, swa_model, device=DEVICE)
        swa_model_path = os.path.join(caminho_checkpoint, "swa_model.pth")
        torch.save({
            'model_state_dict': swa_model.state_dict(),
            'swa_n_averaged': swa_model.n_averaged,
            'best_threshold': best_threshold,
            'best_val_loss': best_val_loss
        }, swa_model_path)
        print(f"💾 Modelo SWA salvo em {swa_model_path}")

        print("\n🔍 Avaliando modelo SWA no conjunto de validação...")
        best_t_swa, _ = buscar_melhor_threshold(swa_model, val_loader, DEVICE)
        iou_swa, prec_swa, rec_swa = avaliar_modelo(swa_model, val_loader, DEVICE, threshold=best_t_swa)
        print(f"SWA Val - IoU: {iou_swa:.4f} | Precisão: {prec_swa:.4f} | Recall: {rec_swa:.4f}")

    if test_loader is not None:
        print("\n🔍 Avaliando modelo normal no conjunto de TESTE...")
        iou_test, prec_test, rec_test = avaliar_modelo(model, test_loader, DEVICE, threshold=best_threshold)
        print(f"Test Normal - IoU: {iou_test:.4f} | Precisão: {prec_test:.4f} | Recall: {rec_test:.4f}")

        if (val_loader is not None) and (start_epoch < swa_start_epoch < epocas):
            print("\n🔍 Avaliando modelo SWA no conjunto de TESTE...")
            best_t_swa_test, _ = buscar_melhor_threshold(swa_model, test_loader, DEVICE)
            iou_swa_test, prec_swa_test, rec_swa_test = avaliar_modelo(swa_model, test_loader, DEVICE, threshold=best_t_swa_test)
            print(f"Test SWA - IoU: {iou_swa_test:.4f} | Precisão: {prec_swa_test:.4f} | Recall: {rec_swa_test:.4f}")

    writer.close()
