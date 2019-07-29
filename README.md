# CVPR_CNN
CVPR_CNN is a simple engineering project of USTCSSE. it is based on another git project [License-Plate-Recognition](https://github.com/wzh191920/License-Plate-Recognition). Developed by python, it could run on the Windows or raspberryOS. It is a simple Chinese Vehicle License Plate Recognition System. if you want to kown more information. Please read /结题材料 or [click here](https://github.com/L-TianRan/CVPR_CNN/tree/master/结题材料)
## Explanation of Files & Directory
There are many .py files in the root directory. I will explain the use of those developed by ourself. As for other files, please read source code of original project [CLPR](https://github.com/wzh191920/License-Plate-Recognition).
### Main File
You can run the whole project by running the `surface.py`. In fact, all you need is `surface.py`, `predict.py`, `label_define.py`, `carplate_CNN_model.h5` and other necessary files of original project.
### Directory
there are 5 directories in the root directory.
`Screenshots` contains some screenshots of the software.
`test` contains many pictures of cars or license plates that could be the input of the software.
`train` contains many images of chars that may appear on the license plate. You can re-train a new network if you want.
`树莓派上所需要的部分.whl文件` contains some .whl files. It could be useful if you want to build the environment in a new raspberryOS.
`结题材料` contains some documents that describe what CVPR_CNN is.

