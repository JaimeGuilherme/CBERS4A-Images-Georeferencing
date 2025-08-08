import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

# Exemplo básico de função para recortar imagem em patches (não obrigatório no seu fluxo)
def patchify_image(img_array, patch_size, stride):
    patches = []
    positions = []
    h, w, _ = img_array.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    return patches, positions

# Reconstruir coordenadas espaciais (x,y) da imagem original a partir do pixel no patch
def reconstruct_coords_from_patch(px, py, transform):
    """
    px, py = coordenadas do pixel na imagem (col, row)
    transform = affine transform rasterio
    """
    x, y = transform * (px, py)
    return x, y

# Carregar modelo salvo do último checkpoint
def load_checkpoint(caminho, model, optimizer=None):
    checkpoint = torch.load(caminho, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_iou = checkpoint.get('best_iou', 0.0)
    return {'epoch': epoch, 'best_iou': best_iou}