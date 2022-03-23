from model import Generator
from utils import *
import torch as t
from torchvision import transforms
from dataset import ImageFolder, IMG_EXTENSIONS
from torch.utils.data import DataLoader


def test():
    ###############
    # load config #
    ###############
    args = load_config("config.yaml")

    ###############
    # load models #
    ###############
    _, model_path = find_latest_model(args["result_dir"], args["dataset"])
    if model_path is None:
        raise FileNotFoundError(
            "There is no model file in the directory, is there no training done?")
    params = t.load(model_path)

    # device
    device = t.device("cuda" if args["cuda"]
                      and t.cuda.is_available() else "cpu")

    # Generator
    genA2B = Generator(input_nc=3,
                       output_nc=3,
                       n_hiddens=args["ch"],
                       n_resblocks=4,
                       img_size=args["img_size"],
                       light=args["light"]).to(device)
    genB2A = Generator(input_nc=3,
                       output_nc=3,
                       n_hiddens=args["ch"],
                       n_resblocks=4,
                       img_size=args["img_size"],
                       light=args["light"]).to(device)
    # load state_dict
    genA2B.load_state_dict(params["genA2B"])
    genB2A.load_state_dict(params["genB2A"])
    print(f"load {model_path} SUCCESS")

    ########################
    # Prepare test dataset #
    ########################
    test_transform = transforms.Compose([
        transforms.Resize((args["img_size"], args["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])
    testA = ImageFolder(os.path.join(
        "datasets", args["dataset"], "testA"), IMG_EXTENSIONS, transform=test_transform)
    testB = ImageFolder(os.path.join(
        "datasets", args["dataset"], "testB"), IMG_EXTENSIONS, transform=test_transform)
    testA_loader = DataLoader(testA, batch_size=1, shuffle=False)
    testB_loader = DataLoader(testB, batch_size=1, shuffle=False)

    ##############
    # test start #
    ##############
    genA2B.eval(), genB2A.eval()

    for n, (real_A, _) in enumerate(testA_loader):
        real_A = real_A.to(device)
        # A2B
        fake_A2B, _, fake_A2B_heatmap = genA2B.forward(real_A)
        # Reconstruction
        fake_A2B2A, _, fake_A2B2A_heatmap = genB2A.forward(fake_A2B)
        # Identity
        fake_A2A, _, fake_A2A_heatmap = genB2A.forward(real_A)

        A2B = np.concatenate(
            (
                handle_generated_image(real_A[0]),

                handle_cam_heatmap(fake_A2B_heatmap[0], size=args["img_size"]),
                handle_generated_image(fake_A2B[0]),

                handle_cam_heatmap(
                    fake_A2B2A_heatmap[0], size=args["img_size"]),
                handle_generated_image(fake_A2B2A[0]),

                handle_cam_heatmap(fake_A2A_heatmap[0], size=args["img_size"]),
                handle_generated_image(fake_A2A[0]),
            ),
            axis=0
        )

        cv2.imwrite(os.path.join(
            args["result_dir"], args["dataset"], "test", "A2B_%d.png" % (n+1)), A2B*255.)

    for n, (real_B, _) in enumerate(testB_loader):
        real_B = real_B.to(device)
        # B2A
        fake_B2A, _, fake_B2A_heatmap = genB2A.forward(real_B)
        # Reconstruction
        fake_B2A2B, _, fake_B2A2B_heatmap = genA2B.forward(fake_B2A)
        # Identity
        fake_B2B, _, fake_B2B_heatmap = genA2B.forward(real_B)

        B2A = np.concatenate(
            (
                handle_generated_image(real_B[0]),

                handle_cam_heatmap(fake_B2A_heatmap[0], size=args["img_size"]),
                handle_generated_image(fake_B2A[0]),

                handle_cam_heatmap(
                    fake_B2A2B_heatmap[0], size=args["img_size"]),
                handle_generated_image(fake_B2A2B[0]),

                handle_cam_heatmap(fake_B2B_heatmap[0], size=args["img_size"]),
                handle_generated_image(fake_B2B[0]),
            ),
            axis=0
        )

        cv2.imwrite(os.path.join(
            args["result_dir"], args["dataset"], "test", "B2A_%d.png" % (n+1)), B2A*255.)


if __name__ == "__main__":
    test()
