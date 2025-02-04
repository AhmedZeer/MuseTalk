export FFMPEG_PATH=/workspace/miniconda3/envs/MuseTalk/bin/ffmpeg
echo $FFMPEG_PATH

pip install -r requirements.txt
huggingface-cli download TMElyralab/MuseTalk --local-dir ./models/
huggingface-cli download stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin config.json --local-dir ./models/sd-vae-ft-mse 
huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir ./models/dwpose 
huggingface-cli download ManyOtherFunctions/face-parse-bisent 79999_iter.pth resnet18-5c106cde.pth --local-dir ./models/face-parse-bisent

pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 

(
  mkdir ./models/whisper
    cd ./models/whisper
      wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
)

cp ./dynamic_modules_utils.py  /usr/local/lib/python3.11/dist-packages/diffusers/utils/dynamic_modules_utils.py
