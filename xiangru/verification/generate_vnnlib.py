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


def generate_csv(args, relpath=False):
    fname = args.output_filename + ".csv"
    with open(fname, "w") as out:
        print(f"Generating {fname}")
        if relpath:
            out.write(f"{os.path.basename(args.output_filename)}.vnnlib\n")
        else:
            out.write(f"{os.getcwd()}/{args.output_filename}.vnnlib\n")
    print(f"Done. Now change your verification config file to verify {fname}.")


def prepare_input_bounds(image):
    image = cv2.resize(image, (128, 128))
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
        "-r",
        "--relative_vnnlib_path",
        action="store_true",
        help="When specified, the vnnlib file path in CSV file will be relative to the path of the CSV file.",
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
    lower_limit = prepare_input_bounds(input_bounds["image_lb"]).flatten()
    upper_limit = prepare_input_bounds(input_bounds["image_ub"]).flatten()
    upper_limit = torch.max(upper_limit, lower_limit)
    lower_limit = lower_limit.tolist()
    upper_limit = upper_limit.tolist()
    state_dim = len(lower_limit)

    fname = f"{args.output_filename}.vnnlib"
    with open(fname, "w") as out:
        print(f"Generating {fname} with")
        generate_preamble(out, state_dim, args)
        generate_limits(out, lower_limit, upper_limit)
        generate_specs(out, args)
    generate_csv(args, relpath=args.relative_vnnlib_path)


if __name__ == "__main__":
    main()