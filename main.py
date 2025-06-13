from Train import train_model

def main():
    img_dir = "/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/images/train"
    label_dir = "/home/francesca/Scrivania/ResNetBBoxRegressor/Dataset/labels/train"
    
    train_model(img_dir, label_dir)

if __name__ == "__main__":
    main()
#     for i, (img, target_bbox) in enumerate(dataloader):
#         with torch.no_grad():
#             pred = model(img.to(device))
#         pred_bbox = pred.squeeze().cpu().numpy()
#         print(f"Pred shape: {pred.shape}, pred squeezed: {pred.squeeze().shape}, pred values: {pred.squeeze().numpy()}")
#         print(f"Predizione {i+1}: {pred_bbox}")
#         # Disegna la bbox #         draw = ImageDraw.Draw(img)                                                            