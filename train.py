import torchvision.transforms as transforms
import torch as t
from utils import *
from dataset import IMG_EXTENSIONS, ImageFolder
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import itertools
from model import Generator, Discriminator, RhoClipper
import numpy as np


def train():
    ###############
    # load config #
    ###############
    args = load_config("config.yaml")
    writer = SummaryWriter(os.path.join(
        args["result_dir"], args["dataset"], "logs"))
    #############################################
    # set benchmark flag to accelerate training #
    #############################################
    device = t.device("cuda" if args["cuda"]
                      and t.cuda.is_available() else "cpu")

    if t.backends.cudnn.enabled and args["benchmark_flag"]:
        t.backends.cudnn.benchmark = True

    ################################
    # Build Dataset and Dataloader #
    ################################

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args["img_size"]+30, args["img_size"]+30)),
        transforms.RandomCrop(args["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args["img_size"], args["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    # Dataset
    trainA = ImageFolder(os.path.join(
        "datasets", args["dataset"], "trainA"), IMG_EXTENSIONS, transform=train_transform)
    trainB = ImageFolder(os.path.join(
        "datasets", args["dataset"], "trainB"), IMG_EXTENSIONS, transform=train_transform)
    testA = ImageFolder(os.path.join(
        "datasets", args["dataset"], "testA"), IMG_EXTENSIONS, transform=test_transform)
    testB = ImageFolder(os.path.join(
        "datasets", args["dataset"], "testB"), IMG_EXTENSIONS, transform=test_transform)

    # Dataloader
    trainA_loader = DataLoader(
        trainA, batch_size=args["batch_size"], shuffle=True)
    trainB_loader = DataLoader(
        trainB, batch_size=args["batch_size"], shuffle=True)
    testA_loader = DataLoader(testA, batch_size=1, shuffle=False)
    testB_loader = DataLoader(testB, batch_size=1, shuffle=False)

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

    # Global Discriminator
    disGA = add_spectral_norm(Discriminator(input_ch=3,
                                            n_hiddens=args["ch"],
                                            n_layers=7).to(device))
    disGB = add_spectral_norm(Discriminator(input_ch=3,
                                            n_hiddens=args["ch"],
                                            n_layers=7).to(device))

    # Local Discriminator
    disLA = add_spectral_norm(Discriminator(input_ch=3,
                                            n_hiddens=args["ch"],
                                            n_layers=5).to(device))
    disLB = add_spectral_norm(Discriminator(input_ch=3,
                                            n_hiddens=args["ch"],
                                            n_layers=5).to(device))

    # Loss Function
    L1_loss = nn.L1Loss().to(device)
    MSE_loss = nn.MSELoss().to(device)
    BCE_loss = nn.BCEWithLogitsLoss().to(device)

    # Optimizer
    optim_gen = t.optim.Adam(itertools.chain(genB2A.parameters(), genA2B.parameters()),
                             lr=args["lr"],
                             betas=(0.5, 0.999),
                             weight_decay=args["weight_decay"])
    optim_dis = t.optim.Adam(itertools.chain(disGA.parameters(), disGB.parameters(), disLA.parameters(), disLB.parameters()),
                             lr=args["lr"],
                             betas=(0.5, 0.999),
                             weight_decay=args["weight_decay"])

    # RhoClippler
    rho_clipper = RhoClipper(0, 1)

    """---------------------------------------------------------------------------"""

    ####################
    # Training Process #
    ####################

    # switch to train mode
    genA2B.train(), genB2A.train(), disGA.train(
    ), disGB.train(), disLA.train(), disLB.train()

    start_iter = 1
    # load model if resume training flag is set
    if args["resume"]:
        start_iter, model_path = find_latest_model(
            args["result_dir"], args["dataset"])
        # load the model
        if start_iter > 0:
            params = t.load(model_path)
            genA2B.load_state_dict(params["genA2B"])
            genB2A.load_state_dict(params["genB2A"])
            disGA.load_state_dict(params["disGA"])
            disGB.load_state_dict(params["disGB"])
            disLA.load_state_dict(params["genA2B"])
            disLB.load_state_dict(params["genA2B"])
            print(f"load {model_path} SUCCESS")
            # adjust the learning rate
            if args["decay_flag"] and start_iter > (args["iteration"]//2):
                optim_gen.param_groups[0]["lr"] -= (
                    args["lr"]/(args["iteration"]//2))*(start_iter-args["iteration"]//2)
                optim_dis.param_groups[0]["lr"] -= (
                    args["lr"]/(args["iteration"]//2))*(start_iter-args["iteration"]//2)

    # Training Loop
    print("training start...")
    start_time = time.time()
    for step in range(start_iter, args["iteration"]+1):
        # Reduce learning rate to half
        if args["decay_flag"] and step > (args["iteration"]//2):
            optim_dis.param_groups[0]["lr"] -= (
                args["lr"]/(args["iteration"]//2))
            optim_gen.param_groups[0]["lr"] -= (
                args["lr"]/(args["iteration"]//2))

        # Extract Data from the Dataloader
        try:
            real_A, _ = trainA_iter.next()
        except:
            trainA_iter = iter(trainA_loader)
            real_A, _ = trainA_iter.next()

        try:
            real_B, _ = trainB_iter.next()
        except:
            trainB_iter = iter(trainB_loader)
            real_B, _ = trainB_iter.next()

        # Transfer to cuda device
        real_A, real_B = real_A.to(device), real_B.to(device)

        #########################
        # Update Discriminators #
        #########################
        optim_dis.zero_grad()
        # Get the generated images A->B and B->A
        fake_A2B, _, _ = genA2B.forward(real_A)
        fake_B2A, _, _ = genB2A.forward(real_B)
        # call detach() method to reduce memory consumption
        fake_A2B = fake_A2B.detach()
        fake_B2A = fake_A2B.detach()
        # Get the log probability of the real images
        real_GA_logit, real_GA_cam_logit, _ = disGA.forward(real_A)
        real_LA_logit, real_LA_cam_logit, _ = disLA.forward(real_A)
        real_GB_logit, real_GB_cam_logit, _ = disGB.forward(real_B)
        real_LB_logit, real_LB_cam_logit, _ = disLB.forward(real_B)
        # Get the log probability of the fake images
        fake_GA_logit, fake_GA_cam_logit, _ = disGA.forward(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = disLA.forward(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = disGB.forward(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = disLB.forward(fake_A2B)
        # Compute loss of 4 Discriminator. Each discriminator has two loss functions
        D_ad_loss_GA = MSE_loss(real_GA_logit, t.ones_like(real_GA_logit, device=device)) +\
            MSE_loss(fake_GA_logit, t.zeros_like(fake_GA_logit, device=device))
        D_ad_cam_loss_GA = MSE_loss(real_GA_cam_logit, t.ones_like(real_GA_cam_logit, device=device)) +\
            MSE_loss(fake_GA_cam_logit, t.zeros_like(
                fake_GA_cam_logit, device=device))
        D_ad_loss_LA = MSE_loss(real_LA_logit, t.ones_like(real_LA_logit, device=device)) +\
            MSE_loss(fake_LA_logit, t.zeros_like(fake_LA_logit, device=device))
        D_ad_cam_loss_LA = MSE_loss(real_LA_cam_logit, t.ones_like(real_LA_cam_logit, device=device)) +\
            MSE_loss(fake_LA_cam_logit, t.zeros_like(
                fake_LA_cam_logit, device=device))
        D_ad_loss_GB = MSE_loss(real_GB_logit, t.ones_like(real_GB_logit, device=device)) +\
            MSE_loss(fake_GB_logit, t.zeros_like(fake_GB_logit, device=device))
        D_ad_cam_loss_GB = MSE_loss(real_GB_cam_logit, t.ones_like(real_GB_cam_logit, device=device)) +\
            MSE_loss(fake_GB_cam_logit, t.zeros_like(
                fake_GB_cam_logit, device=device))
        D_ad_loss_LB = MSE_loss(real_LB_logit, t.ones_like(real_LB_logit, device=device)) +\
            MSE_loss(fake_LB_logit, t.zeros_like(fake_LB_logit, device=device))
        D_ad_cam_loss_LB = MSE_loss(real_LB_cam_logit, t.ones_like(real_LB_cam_logit, device=device)) +\
            MSE_loss(fake_LB_cam_logit, t.zeros_like(
                fake_LB_cam_logit, device=device))

        D_loss_A = args["adv_weight"] * \
            (D_ad_loss_GA+D_ad_cam_loss_GA+D_ad_loss_LA+D_ad_cam_loss_LA)
        D_loss_B = args["adv_weight"] * \
            (D_ad_loss_GB+D_ad_cam_loss_GB+D_ad_loss_LB+D_ad_cam_loss_LB)

        Discriminator_loss = D_loss_A+D_loss_B
        Discriminator_loss.backward()
        optim_dis.step()
        # send data to tensorboardX
        writer.add_scalars("discriminator loss",
                           {"adv_loss_GA": D_ad_loss_GA,
                            "adv_cam_loss_GA": D_ad_cam_loss_GA,
                            "adv_loss_GB": D_ad_loss_GB,
                            "adv_cam_loss_GB": D_ad_cam_loss_GB,
                            "adv_loss_LA": D_ad_loss_LA,
                            "adv_cam_loss_LA": D_ad_cam_loss_LA,
                            "adv_loss_LB": D_ad_loss_LB,
                            "adv_cam_loss_LB": D_ad_cam_loss_LB, }, global_step=step
                           )
        #####################
        # Update Generators #
        #####################
        optim_gen.zero_grad()

        # transform to another domain
        fake_A2B, fake_A2B_cam_logit, _ = genA2B.forward(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = genB2A.forward(real_B)
        # reconstruct to the original domain
        fake_A2B2A, _, _ = genB2A.forward(fake_A2B)
        fake_B2A2B, _, _ = genA2B.forward(fake_B2A)
        # we should transfer the style from the same domain
        fake_A2A, fake_A2A_cam_logit, _ = genB2A.forward(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = genA2B.forward(real_B)
        # evaluate the generated images
        fake_GA_logit, fake_GA_cam_logit, _ = disGA.forward(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = disLA.forward(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = disGB.forward(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = disLB.forward(fake_A2B)
        # compute loss of 2 Generator, each Generator has 4 loss functions

        # Adversarial Loss -> the generator should fake the discriminator
        G_ad_loss_GA = MSE_loss(
            fake_GA_logit, t.ones_like(fake_GA_logit, device=device))
        G_ad_cam_loss_GA = MSE_loss(
            fake_GA_cam_logit, t.ones_like(fake_GA_cam_logit, device=device))
        G_ad_loss_LA = MSE_loss(
            fake_LA_logit, t.ones_like(fake_LA_logit, device=device))
        G_ad_cam_loss_LA = MSE_loss(
            fake_LA_cam_logit, t.ones_like(fake_LA_cam_logit, device=device))
        G_ad_loss_GB = MSE_loss(
            fake_GB_logit, t.ones_like(fake_GB_logit, device=device))
        G_ad_cam_loss_GB = MSE_loss(
            fake_GB_cam_logit, t.ones_like(fake_GB_cam_logit, device=device))
        G_ad_loss_LB = MSE_loss(
            fake_LB_logit, t.ones_like(fake_LB_logit, device=device))
        G_ad_cam_loss_LB = MSE_loss(
            fake_LB_cam_logit, t.ones_like(fake_LB_cam_logit, device=device))
        # conbine the adversarial loss together
        G_ad_loss_A = G_ad_loss_GA+G_ad_cam_loss_GA+G_ad_loss_LA+G_ad_cam_loss_LA
        G_ad_loss_B = G_ad_loss_GB+G_ad_cam_loss_GB+G_ad_loss_LB+G_ad_cam_loss_LB
        # Reconstruction Loss, in the process A->B->A' ,A and A' should be same
        G_recon_loss_A = L1_loss(fake_A2B2A, real_A)
        G_recon_loss_B = L1_loss(fake_B2A2B, real_B)
        # Identity loss, the generator should not change the image from target domain
        G_identity_loss_A = L1_loss(fake_A2A, real_A)
        G_identity_loss_B = L1_loss(fake_B2B, real_B)
        # Class Activation Mapping loss -> The auxiliary classifier should distinguish
        # whether the image is from the source or target domain
        G_cam_loss_A = BCE_loss(fake_B2A_cam_logit, t.ones_like(fake_B2A_cam_logit, device=device)) +\
            BCE_loss(fake_A2A_cam_logit, t.zeros_like(
                fake_A2A_cam_logit, device=device))
        G_cam_loss_B = BCE_loss(fake_A2B_cam_logit, t.ones_like(fake_A2B_cam_logit, device=device)) +\
            BCE_loss(fake_B2B_cam_logit, t.zeros_like(
                fake_B2B_cam_logit, device=device))

        # Combine them together
        G_loss_A = args["adv_weight"]*G_ad_loss_A+args["cycle_weight"]*G_recon_loss_A + \
            args["identity_weight"]*G_identity_loss_A + \
            args["cam_weight"]*G_cam_loss_A
        G_loss_B = args["adv_weight"]*G_ad_loss_B+args["cycle_weight"]*G_recon_loss_B + \
            args["identity_weight"]*G_identity_loss_B + \
            args["cam_weight"]*G_cam_loss_B

        Generator_loss = G_loss_A+G_loss_B
        Generator_loss.backward()
        optim_gen.step()
        # send data to tensorboardX
        writer.add_scalars("generator_loss",
                           {"adv_loss_A": G_ad_loss_A*args["adv_weight"],
                            "adv_loss_B": G_ad_loss_B*args["adv_weight"],
                            "recon_loss_A": G_recon_loss_A*args["cycle_weight"],
                            "recon_loss_B": G_recon_loss_B*args["cycle_weight"],
                            "idt_loss_A": G_identity_loss_A*args["identity_weight"],
                            "idt_loss_B": G_identity_loss_B*args["identity_weight"],
                            "cam_loss_A": G_cam_loss_A*args["cam_weight"],
                            "cam_loss_B": G_cam_loss_B*args["cam_weight"],
                            }, global_step=step)
        ##################################################
        # Clip the parameter of the AdaILN and ILN layer #
        ##################################################
        genA2B.apply(rho_clipper)
        genB2A.apply(rho_clipper)

        print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step,
                                                                    args["iteration"], time.time() - start_time, Discriminator_loss, Generator_loss))

        ######################
        # Validation Process #
        ######################
        with t.no_grad():
            if step % args["print_freq"] == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((args["img_size"]*7, 0, 3))
                B2A = np.zeros((args["img_size"]*7, 0, 3))
                # turn to eval mode
                genA2B.eval(), genB2A.eval()
                # generate train set images
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = trainA_iter.next()
                    except:
                        trainA_iter = iter(trainA_loader)
                        real_A, _ = trainA_iter.next()
                    try:
                        real_B, _ = trainB_iter.next()
                    except:
                        trainB_iter = iter(trainB_loader)
                        real_B, _ = trainB_iter.next()

                    real_A, real_B = real_A.to(device), real_B.to(device)
                    # A to B -> Style Transform
                    fake_A2B, _, fake_A2B_heatmap = genA2B.forward(real_A)
                    fake_B2A, _, fake_B2A_heatmap = genB2A.forward(real_B)
                    # A to B to A -> Reconstruction
                    fake_A2B2A, _, fake_A2B2A_heatmap = genB2A.forward(
                        fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = genA2B.forward(
                        fake_B2A)
                    # A to A -> Identity
                    fake_A2A, _, fake_A2A_heatmap = genB2A.forward(real_A)
                    fake_B2B, _, fake_B2B_heatmap = genA2B.forward(real_B)
                    # concatenate
                    A2B = np.concatenate((A2B, np.concatenate((
                                         handle_generated_image(real_A[0]),
                                         handle_cam_heatmap(
                                             fake_A2A_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_A2A[0]),
                                         handle_cam_heatmap(
                                             fake_A2B_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_A2B[0]),
                                         handle_cam_heatmap(
                                             fake_A2B2A_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_A2B2A[0])), axis=0)), axis=1)

                    B2A = np.concatenate((B2A, np.concatenate((
                                         handle_generated_image(real_B[0]),
                                         handle_cam_heatmap(
                                             fake_B2B_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_B2B[0]),
                                         handle_cam_heatmap(
                                             fake_B2A_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_B2A[0]),
                                         handle_cam_heatmap(
                                             fake_B2A2B_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_B2A2B[0])), axis=0)), axis=1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(testA_loader)
                        real_A, _ = testA_iter.next()
                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(testB_loader)
                        real_B, _ = testB_iter.next()

                    real_A, real_B = real_A.to(device), real_B.to(device)
                    # A to B -> Style Transform
                    fake_A2B, _, fake_A2B_heatmap = genA2B.forward(real_A)
                    fake_B2A, _, fake_B2A_heatmap = genB2A.forward(real_B)
                    # A to B to A -> Reconstruction
                    fake_A2B2A, _, fake_A2B2A_heatmap = genB2A.forward(
                        fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = genA2B.forward(
                        fake_B2A)
                    # A to A -> Identity
                    fake_A2A, _, fake_A2A_heatmap = genB2A.forward(real_A)
                    fake_B2B, _, fake_B2B_heatmap = genA2B.forward(real_B)
                    # concatenate
                    A2B = np.concatenate((A2B, np.concatenate((
                                         handle_generated_image(real_A[0]),
                                         handle_cam_heatmap(
                                             fake_A2A_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_A2A[0]),
                                         handle_cam_heatmap(
                                             fake_A2B_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_A2B[0]),
                                         handle_cam_heatmap(
                                             fake_A2B2A_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_A2B2A[0])), axis=0)), axis=1)

                    B2A = np.concatenate((B2A, np.concatenate((
                                         handle_generated_image(real_B[0]),
                                         handle_cam_heatmap(
                                             fake_B2B_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_B2B[0]),
                                         handle_cam_heatmap(
                                             fake_B2A_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_B2A[0]),
                                         handle_cam_heatmap(
                                             fake_B2A2B_heatmap[0], size=args["img_size"]),
                                         handle_generated_image(fake_B2A2B[0])), axis=0)), axis=1)

                # write image
                cv2.imwrite(os.path.join(
                    args["result_dir"], args["dataset"], "img", "A2B_%07d.png" % step), A2B*255.0)
                cv2.imwrite(os.path.join(
                    args["result_dir"], args["dataset"], "img", "B2A_%07d.png" % step), B2A*255.0)
                # turn to train mode
                genA2B.train(), genB2A.train()

        if step % args["save_freq"] == 0:
            params = {}
            params['genA2B'] = genA2B.state_dict()
            params['genB2A'] = genB2A.state_dict()
            params['disGA'] = disGA.state_dict()
            params['disGB'] = disGB.state_dict()
            params['disLA'] = disLA.state_dict()
            params['disLB'] = disLB.state_dict()
            t.save(params, os.path.join(args["result_dir"],
                                        args["dataset"] + "model", '_params_%07d.pt' % step))


if __name__ == "__main__":
    train()
