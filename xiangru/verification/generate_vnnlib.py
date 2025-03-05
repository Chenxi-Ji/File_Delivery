import os
import sys
import time
import socket
import argparse
import numpy as np
import torch
import cv2


def generate_preamble(out, state_dim, args):
    out.write("; VNNLIB Property for the verification of Lyapunov condition.\n\n")
    try:
        user = os.getlogin()
    except OSError:
        user = "UNKNOWN_USER"
    out.write(
        f"; Generated at {time.ctime()} on {socket.gethostname()} by {user}\n"
    )
    out.write(f'; Generation command: \n; {" ".join(sys.argv)}\n\n')

    for i in range(state_dim):
        out.write(f"(declare-const X_{i} Real)\n")
    for i in range(args.num_classes):
        out.write(f"(declare-const Y_{i} Real)\n")

    out.write("\n")


def generate_limits(out, lower_limit, upper_limit):
    assert len(lower_limit) == len(upper_limit)
    out.write("; Input constraints.\n\n")
    for i, (l, u) in enumerate(zip(lower_limit, upper_limit)):
        out.write(f"(assert (<= X_{i} {u}))\n")
        out.write(f"(assert (>= X_{i} {l}))\n\n")


def generate_specs(out, args):
    out.write("\n; Specifications.\n\n")
    target_class = args.target_class
    out.write(f"(assert (or\n")
    for i in range(args.num_classes):
        if i != target_class:
            out.write(f"  (and (<= Y_{target_class} Y_{i}))\n")
    out.write("))\n\n")


def generate_csv(args, num_specs):
    fname = "specs/" + args.output_filename + ".csv"
    with open(fname, "w") as out:
        print(f"Generating {fname}")
        for i in range(num_specs):
            out.write(f"{os.getcwd()}/specs/{args.output_filename}_{i}.vnnlib\n")
    print(f"Done. Now change your verification config file to verify {fname}.")


def prepare_input_bounds(image):
    # image: (H, W, C) or (N, H, W, C)
    # If image is in batch, assume it doesn't need to be resized.
    if image.ndim == 4:
        image = torch.Tensor(image)
        image = image.permute(0, 3, 1, 2)
    else:
        image = cv2.resize(image, (25, 25))
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def main():
    parser = argparse.ArgumentParser(
        prog="VNNLIB Generator",
        description="Generate VNNLIB property file for verification of nerf classification",
    )
    parser.add_argument(
        "output_filename",
        type=str,
        help="Output filename prefix. A single csv file and multiple VNNLIB files will be generated.",
    )
    parser.add_argument(
        "-b",
        "--bound_path",
        type=str,
        default="input_bounds.pth",
        help="Path to the bounding box file.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes.",
    )
    parser.add_argument(
        "-t",
        "--target_class",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    input_bounds = np.load(args.bound_path)
    lower_limit = prepare_input_bounds(input_bounds["images_lb"])
    upper_limit = prepare_input_bounds(input_bounds["images_ub"])
    upper_limit = torch.max(upper_limit, lower_limit)
    state_dim = lower_limit.reshape(lower_limit.size(0), -1).size(1)
    num_specs = lower_limit.size(0)

    for i in range(num_specs):
        fname = f"specs/{args.output_filename}_{i}.vnnlib"
        with open(fname, "w") as out:
            print(f"Generating {fname}")
            lower_limit_i = lower_limit[i].reshape(-1).tolist()
            upper_limit_i = upper_limit[i].reshape(-1).tolist()
            generate_preamble(out, state_dim, args)
            generate_limits(out, lower_limit_i, upper_limit_i)
            generate_specs(out, args)
    generate_csv(args, num_specs)


if __name__ == "__main__":
    main()