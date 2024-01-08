#!/usr/bin/env python3
import os, sys
import launch
import git
import pkg_resources
import os
import io
import json
from pathlib import Path
from typing import Tuple, Optional
from modules import scripts

import argparse

import modules.scripts as scripts
from modules import script_callbacks
from modules.shared import opts
from modules.paths import models_path

from basicsr.utils.download_util import load_file_from_url


CI_VERSION="0.0.1a"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

root_dir = Path(scripts.basedir()).parent.parent
req_file = os.path.join(Path(BASE_PATH), "requirements.txt")
kohya_path = os.path.join(Path(BASE_PATH), "kohya")

kohya_git_repo_path="https://github.com/bmaltais/kohya_ss"

if not os.path.exists(kohya_path):
    os.makedirs(kohya_path)

repo_dir = os.listdir(kohya_path)

sys.path.append(kohya_path)

def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    """ trimesh[easy] -> trimesh """
    # Split the string on '[' and take the first part
    base_package = package_string.split('[')[0]
    return base_package

def install_requirements(req_file):
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if "==" in package:
                    package_name, package_version = package.split("==")
                    installed_version = get_installed_version(package_name)
                    if installed_version != package_version:
                        launch.run_pip(
                            f"install -U {package}",
                            f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif ">=" in package:
                    package_name, package_version = package.split(">=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or comparable_version(
                        installed_version
                    ) < comparable_version(package_version):
                        launch.run_pip(
                            f"install -U {package}",
                            f"kohya_embedded requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif not launch.is_installed(extract_base_package(package)):
                    launch.run_pip(
                        f"install {package}",
                        f"kohya_embedded requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, some parts of KohyaSS may not work."
                )

if (len(repo_dir) == 0):
    print("Installing Kohya.")
    # Firstly we install necessary libraries
    install_requirements(req_file)
    # Next, we clone Kohya repository
    if len(repo_dir) == 0: # Checking if there is nothing in Kohya's directory.
        git.Repo.clone_from(kohya_git_repo_path, kohya_path)
    # And last but not least, we make symlinks for the necessary scripts
    symlinks_required = [
	"train_textual_inversion.py",
	"train_db.py",
	"train_network.py",
	"train_controlnet.py",
	"sdxl_train.py",
	"train_textual_inversion_XTI.py",
	"sdxl_train_control_net_lllite.py",
	"sdxl_train_network.py",
	"sdxl_train_textual_inversion.py"
    ]
    for sname in symlinks_required:
        if not os.path.isfile(os.path.join(root_dir,sname)):
            os.symlink(os.path.join(kohya_path,sname) , os.path.join(root_dir,sname))
    print("Kohya installation complete.")

