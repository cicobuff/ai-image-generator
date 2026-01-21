# AI Image Generator

## Introduction
This is a project that creates an app for Local Image Generation using a local llm.
The app is fully local and will not depend on external models and API for image generation.

## Platform

* the project will be built for linux environments with gtk 4
* language will be python version 3.12
* graphics library will be Cairo
* gui library will be gtk 4.0
* text rendering will be pillow

## GUI Design

* the UX flow consist of the following screens
  * there will be an initial setup_screen where the following configuration is set and saved
    * the settings directory is defaulted to ~/.aiimagegen, the file is a config.yaml
    * the output directory is defaulted to ./output relatively to the app directory
    * the models directory where any diffusion models are stored, this is defaulted to ./models relatively to the app directory
    * the setup_screen is only displayed when the settings directory and the config.yaml is not present
    * at first launch, the app should check for the number of gpu present and show a list of the gpus as checkbox
      * when the user select multiple gpus, the config should be saved and the app will use the selected gpus only
  * upon launch, the work_screen will be displayed with the following layout
    * a menubar at the top
    * a toolbar below the menubar
      * there should be a load model button that will load all the models selected in the left panel
      * there should be a clear model button that will unload all the models selected
      * there should be a generate image button that will generate
    * a left panel 
      * at the top of the left panel there should be a model selector, there should be a display of the vram of the models that are loaded
        * the vram usage should show how much the checkpoint, vae and clip are in the gpu's ram
        * there should be a total vram and used and remaining vram display
        * the usage breakdown should be grouped by the different gpu if more than 1 gpu exist
        * there should be a dropdown selection for the llm model to be used for generation
        * there should be a dropdown selection for the vae
        * there should be a dropdown selection for the clip
      * there is a generation param area that contains the following
        * image size selector, defaulted to 1024 by 1024, can be saved into settings
        * sampler selector, defaulted to Euler
        * scheduler selection, defaulted to normal, can be saved into settings
        * steps defaulted to 20, can be saved into settings
        * cfg scale defaulted to 7, can be saved into settings
        * seed defaulted to -1 which is random
    * a center panel
      * the center panel consists of a large image display area
      * below the image display area, there is a prompt area that has 2 text boxes for prompts, 
        * the positive prompt textbox should have a green border
        * the negative prompt textbox should have a red border
    * a right panel
      * the right panel consists of a scrollable list of images that have been generated so far, they are thumbnails display
      
* models folder structure
  * there should be a checkpoints directory where the diffusion models are stored
  * there should be a vae directory where the vae models are stored
  * there should be a clip directory where the clip models are stored

* models handling
  * some diffusion models in checkpoints directory are all-in-one models that have vae and clip built in, when these types of models are selected, the vae and clip should use the embedded ones unless specifically overridden by vae and clip selectors

* image generation
  * the models are stable diffusion
    * for image generation, the app should try to load all the models into vram and the perform the image generation

* there may be more than 1 gpu present, in my system, i have 2 gpus with NVLink 
  * if possible, make the code leverage the 2 gpus
    * load the checkpoint and vae and clip into different gpu is necessary