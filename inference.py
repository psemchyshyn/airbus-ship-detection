from model import ShipSegmentModel


def segment_images(folder, out, checkpoint_path):
    model = ShipSegmentModel.load_from_checkpoint(checkpoint_path).cuda()
    model.predict(folder, out)


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    folder = "/mnt/data/psemchyshyn/ship-detection-data/test_v2"
    out = "/mnt/data/psemchyshyn/ship-detection-data/test_prediction"
    checkpoint_path = "/mnt/data/psemchyshyn/checkpoints/ship-detection/unet34_sparse_augs_ce_focal_lovasz_loss_balanced_reduced/last.ckpt"
    segment_images(folder, out, checkpoint_path)
