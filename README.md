# kohya-with-automatic1111
The unification of Kohya_SS and Automatic1111 Stable Diffusion WebUI

(Currently Linux with Nvidia GPU only)
The file sd_requirements.txt is the requirements.txt file for pip which currently works for BOTH kohya_ss and automatic1111 stable diffusion webui project.
How to use?
1. `cd <your_a1111_folder>`
2. `source venv/bin/activate`
3. `pip install -r <where_you_have_downloaded_file_sd_requirements_txt>`
4. edit webui-user.sh - e.g. `nano webui-user.sh`
5. add "--skip-install" in to "export COMMANDLINE_ARGS" (uncomment it for sure), e.g. `export COMMANDLINE_ARGS="--skip-install"`
6. we're done with a1111
7. now, kohya
8. `cd <your_kohya_folder>`
9. `source venv/bin/activate`
10. `pip install -r <where_you_have_downloaded_file_sd_requirements_txt>`
11. `mv requirements.txt requirements.txt.backup`
12. `touch requirements.txt`
13. `mv requirements_linux.txt requirements_linux.txt.backup`
14. we're done with kohya
