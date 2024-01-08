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
from modules import scripts, ui_components

import numpy as np
import cv2

import gradio as gr
import argparse

import modules.scripts as scripts
from modules import script_callbacks, shared
from modules.shared import opts
from modules.paths import models_path

from basicsr.utils.download_util import load_file_from_url

from modules.ui import setup_progressbar

from datetime import datetime

#Default values

kohya_git_repo_path_default = "https://github.com/bmaltais/kohya_ss"
kohya_interface_tab_name_default = "KohyaSS"
kohya_show_dreambooth_tab = True
kohya_show_lora_tab = True
kohya_show_ti_tab = True
kohya_show_finetuning_tab = True
kohya_show_utilities_tab = True
kohya_show_about_tab = True
kohya_show_service_tab = True
p_category = "training"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

root_dir = Path(BASE_PATH).parent.parent
req_file = os.path.join(Path(BASE_PATH).parent, "requirements.txt")
kohya_path = os.path.join(Path(BASE_PATH).parent, "kohya")


if not os.path.exists(kohya_path):
    os.makedirs(kohya_path)


kohya_git_repo_path          = shared.opts.data.get("kohya_git_repo_path",         kohya_git_repo_path_default)
kohya_interface_tab_name     = shared.opts.data.get("kohya_interface_tab_name",    kohya_interface_tab_name_default)
#Kohya interface tabs
kohya_show_dreambooth_tab    = shared.opts.data.get("kohya_show_dreambooth_tab",   kohya_show_dreambooth_tab)
kohya_show_lora_tab          = shared.opts.data.get("kohya_show_lora_tab",         kohya_show_lora_tab)
kohya_show_ti_tab            = shared.opts.data.get("kohya_show_ti_tab",           kohya_show_ti_tab)
kohya_show_finetuning_tab    = shared.opts.data.get("kohya_show_finetuning_tab",   kohya_show_finetuning_tab)
kohya_show_utilities_tab     = shared.opts.data.get("kohya_show_utilities_tab",    kohya_show_utilities_tab)
kohya_show_about_tab         = shared.opts.data.get("kohya_show_about_tab",        kohya_show_about_tab)
kohya_show_service_tab       = shared.opts.data.get("kohya_show_service_tab",      kohya_show_service_tab)

def get_kohya_tab_names():
    return ["dreambooth",
        "lora"
        "texturalinversion",
        "finetuning",
        "utilities",
        "about"
        ]

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
                            f"kohya_embedded requirement: changing {package_name} version from {installed_version} to {package_version}",
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
                    f"Warning: Failed to install {package}, some parts of Kohya_SS may not work."
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
    kohya_interface_tab_name_default = "KohyaSS"
    kohya_interface_tab_name = shared.opts.data.get("kohya_interface_tab_name", kohya_interface_tab_name_default)
    return kohya_interface_tab_name

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

js = "(x) => confirm('Are you sure?')"

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
        if kohya_show_dreambooth_tab:
            with gr.Tab("Dreambooth"):
                (
                    train_data_dir_input,
                    reg_data_dir_input,
                    output_dir_input,
                    logging_dir_input,
                ) = dreambooth_tab(headless=headless)
        else:
            train_data_dir_input = None
            reg_data_dir_input = None
            output_dir_input = None
            logging_dir_input = None
        if kohya_show_lora_tab:
            with gr.Tab("LoRA"):
                 lora_tab(headless=headless)
        if kohya_show_ti_tab:
            with gr.Tab("Textual Inversion"):
                ti_tab(headless=headless)
        if kohya_show_finetuning_tab:
            with gr.Tab("Finetuning"):
                finetune_tab(headless=headless)
        if kohya_show_utilities_tab:
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
        if kohya_show_about_tab:
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
        if kohya_show_service_tab:
            with gr.Tab("Service"):
                with gr.Blocks(analytics_enabled=False) as ui_component:
                    with gr.Row():
                        hidden_checkbox = gr.Checkbox(visible=False)
                        btn = gr.Button("Update Kohya_SS repositorium (not the whole plugin)", elem_id="update_kohya_repo", variant='primary')
                    with gr.Row():
                        textbox = gr.Textbox(label="Progress")
                        num = gr.Number(visible=False)
            def upd_repo(checkbox_state, number):
                if checkbox_state:
                    number += 1
                today_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                err_txt = ''
                print("Updating kohya repositorium...")
                try:
                    repo = git.Repo(kohya_path)
                    o = repo.remotes.origin
                    o.pull()
                except Exception as e:
                    err_txt = "\nThere was an error:" + str(e)
                print("Kohya repositorium update done.")
                today_finish = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                display = f'{today_start} - {today_finish} Kohya repo updated ({number:.0f} times).{err_txt}'
                return False, display, number
            btn.click(None, None, hidden_checkbox, _js=js)
            hidden_checkbox.change(upd_repo, [hidden_checkbox, num], [hidden_checkbox, textbox, num])
    os.chdir(scripts.basedir())
    return [(kohya_embedded, kohya_interface_tab_name, "kohya_embedded")]


def on_ui_settings():
    section = ('training', "KohyaSS")
    shared.opts.add_option(
        "kohya_git_repo_path",
        shared.OptionInfo(
            kohya_git_repo_path,
            "Kohya's Git repository URL",
            section=section, category_id=p_category
            )
        )
    shared.opts.add_option(
        "kohya_interface_tab_name",
        shared.OptionInfo(
            kohya_interface_tab_name,
            "The KohyaSS tab name in interface",
            section=section, category_id=p_category
            ).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_dreambooth_tab",
        shared.OptionInfo(
            kohya_show_dreambooth_tab,
            "Show Dreambooth tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_lora_tab",
        shared.OptionInfo(
            kohya_show_lora_tab,
            "Show LoRA tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_ti_tab",
        shared.OptionInfo(
            kohya_show_ti_tab,
            "Show Textual Inversion tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_finetuning_tab",
        shared.OptionInfo(
            kohya_show_finetuning_tab,
            "Show Finetuning tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_utilities_tab",
        shared.OptionInfo(
            kohya_show_utilities_tab,
            "Show Utilities tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_about_tab",
        shared.OptionInfo(
            kohya_show_about_tab,
            "Show About tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    shared.opts.add_option(
        "kohya_show_service_tab",
        shared.OptionInfo(
            kohya_show_service_tab,
            "Show Service tab",
            gr.Checkbox, {"interactive": True}, section=section, category_id=p_category).needs_reload_ui()
        )
    #shared.opts.add_option(
    #    "kohya_ui_tab_order",
    #    shared.OptionInfo(
    #            [],
    #            "Tab order",
    #            ui_components.DropdownMulti,
    #            lambda:{"choices":list(get_kohya_tab_names())}
    #        ).needs_reload_ui()
    #    )


if module_installed:
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_ui_settings(on_ui_settings)
