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

import numpy as np
import cv2

import gradio as gr
import argparse

import modules.scripts as scripts
from modules import script_callbacks
from modules.shared import opts
from modules.paths import models_path

from basicsr.utils.download_util import load_file_from_url


CI_VERSION="0.0.1a"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

root_dir = Path(scripts.basedir()).parent.parent.parent
req_file = os.path.join(Path(BASE_PATH).parent, "requirements.txt")
kohya_path = os.path.join(Path(BASE_PATH).parent, "kohya")

kohya_git_repo_path="https://github.com/bmaltais/kohya_ss"

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



module_installed = True

try:
    from kohya.dreambooth_gui import dreambooth_tab
    from finetune_gui import finetune_tab
    from textual_inversion_gui import ti_tab
    from library.utilities import utilities_tab
    from lora_gui import lora_tab
    from library.class_lora_tab import LoRATools

    from library.custom_logging import setup_logging
    from library.localization_ext import add_javascript

    # Set up logging
    log = setup_logging()

    import matplotlib
    matplotlib.use('Agg')

except ModuleNotFoundError:
    # not (yet?) installed.
    module_installed = False

#export DISPLAY=:0.0
os.environ["DISPLAY"] = ":0.0"

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "Kohya_SS"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

def on_ui_tabs():
    headless = False
    os.chdir(kohya_path)
    release = "N/A"
    if os.path.exists("./.release"):
        with open(os.path.join("./.release"), "r", encoding="utf8") as file:
            release = file.read()
    if os.path.exists("./README.md"):
       	with open(os.path.join("./README.md"), "r", encoding="utf8") as file:
            README = file.read()
    with gr.Blocks(analytics_enabled=False) as kohya_embedded:
        with gr.Tab("Dreambooth"):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(headless=headless)
        with gr.Tab("LoRA"):
             lora_tab(headless=headless)
        with gr.Tab("Textual Inversion"):
            ti_tab(headless=headless)
        with gr.Tab("Finetuning"):
            finetune_tab(headless=headless)
        with gr.Tab("Utilities"):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
               	output_dir_input=output_dir_input,
               	logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )
            with gr.Tab("LoRA"):
                   _ = LoRATools(headless=headless)
        with gr.Tab("About"):
            gr.Markdown(f"kohya_ss GUI release {release}")
            with gr.Tab("README"):
               	gr.Markdown(README)
        htmlStr = f"""
        <html>
            <body>
                <div class="ver-class">{release}</div>
            </body>
        </html>
        """
        gr.HTML(htmlStr)
    os.chdir(scripts.basedir())
    return [(kohya_embedded, "KohyaSS", "kohya_embedded")]

if module_installed:
    script_callbacks.on_ui_tabs(on_ui_tabs)
