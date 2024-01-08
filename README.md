# kohya-with-automatic1111
The unification of Kohya_SS and Automatic1111 Stable Diffusion WebUI

(Currently Linux with Nvidia GPU only)

How to use?
* Install as usual AUTOMATIC1111 plugin.
* Better add "--skip-install" to the `webui-user.sh` file.
* I would recommend also fixing the following part `venv/bin/activate` file like this:
  
  `[...]`
  
  `export PATH`
  
  `export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"`
  
  `export PYTHONWARNINGS="ignore::DeprecationWarning"`
  
  `export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib64:$VIRTUAL_ENV/lib:$VIRTUAL_ENV/lib/python3.11/site-packages/tensorrt/:$VIRTUAL_ENV/lib/python3.11/site-packages/onnxruntime/capi:$VIRTUAL_ENV/lib/python3.11/site-packages/:$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$PATH"`
  
  `export PYTHONPATH="$VIRTUAL_ENV/lib:$VIRTUAL_ENV/lib/python3.11:$VIRTUAL_ENV/lib/python3.11/site-packages"`
  `export PATH="$PATH:$VIRTUAL_ENV/lib:$VIRTUAL_ENV/lib/python3.11:$VIRTUAL_ENV/lib/python3.11/site-packages"`
  
  `[...]`
  (if You're using python not 3.11 - change accordingly.)

~~The file sd_requirements.txt is the requirements.txt file for pip which currently works for BOTH kohya_ss and automatic1111 stable diffusion webui project.~~
~~How to use?~~
~~1. `cd <your_a1111_folder>`~~
~~2. `source venv/bin/activate`~~
~~3. `pip install -r <where_you_have_downloaded_file_sd_requirements_txt>`~~
~~4. edit webui-user.sh - e.g. `nano webui-user.sh`~~
~~5. add "--skip-install" in to "export COMMANDLINE_ARGS" (uncomment it for sure), e.g. `export COMMANDLINE_ARGS="--skip-install"`~~
~~6. we're done with a1111~~
~~7. now, kohya~~
~~8. `cd <your_kohya_folder>`~~
~~9. `source venv/bin/activate`~~
~~10. `pip install -r <where_you_have_downloaded_file_sd_requirements_txt>`~~
~~11. `mv requirements.txt requirements.txt.backup`~~
~~12. `touch requirements.txt`~~
~~13. `mv requirements_linux.txt requirements_linux.txt.backup`~~
~~14. we're done with kohya~~
