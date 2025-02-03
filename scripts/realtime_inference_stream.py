#!/usr/bin/env python
import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
import shutil
import threading
import queue
import time

# Import Flask for the original streaming endpoint.
from flask import Flask, Response, send_file

# Additional imports for WebRTC streaming.
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
import av

# load model weights and helper functions
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending

# Load models and set up devices.
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        # For the original MJPEG streaming:
        self.encoded_frame_queue = queue.Queue()
        # For WebRTC streaming (raw BGR frames):
        self.raw_frame_queue = queue.Queue()
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n) ")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avatar: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()
            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)
            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue) ")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx += 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, total_frames):
        """
        Original method for MJPEG streaming.
        Get frames from res_frame_queue, composite them,
        encode as JPEG, and push them into encoded_frame_queue.
        """
        print("Starting network stream processing for {} frames...".format(total_frames))
        while self.idx < total_frames:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print("Resize error:", e)
                continue
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            ret, buffer = cv2.imencode('.jpg', combine_frame)
            if ret:
                self.encoded_frame_queue.put(buffer.tobytes())
            self.idx += 1
        self.encoded_frame_queue.put(None)

    def process_frames_webrtc(self, res_frame_queue, total_frames):
        """
        New method for WebRTC streaming.
        Get frames from res_frame_queue, composite them,
        and push the raw (BGR) composite frames into raw_frame_queue.
        """
        print("Starting WebRTC frame processing for {} frames...".format(total_frames))
        while self.idx < total_frames:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print("Resize error:", e)
                continue
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            self.raw_frame_queue.put(combine_frame)
            self.idx += 1
        self.raw_frame_queue.put(None)

    def inference(self, audio_path, fps):
        """
        Original inference method for MJPEG streaming.
        Processes audio and frames; pushes JPEG frames into encoded_frame_queue.
        """
        print("Starting inference...")
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print("Audio processing for {} took {:.2f}ms".format(audio_path, (time.time() - start_time) * 1000))
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        stream_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num))
        stream_thread.start()
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(video_num / self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        stream_thread.join()
        print("Inference and streaming processing completed in {:.2f} seconds".format(time.time() - start_time))

    def inference_webrtc(self, audio_path, fps):
        """
        New inference method for WebRTC streaming.
        Processes audio and frames; pushes raw composite frames into raw_frame_queue.
        """
        print("Starting inference for WebRTC...")
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print("Audio processing for {} took {:.2f}ms".format(audio_path, (time.time() - start_time) * 1000))
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        stream_thread = threading.Thread(target=self.process_frames_webrtc, args=(res_frame_queue, video_num))
        stream_thread.start()
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(video_num / self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        stream_thread.join()
        print("Inference for WebRTC completed in {:.2f} seconds".format(time.time() - start_time))

# ----- Original Flask endpoints for MJPEG + separate audio streaming -----
app = Flask(__name__)
global_avatar = None
global_audio_path = None

@app.route('/')
def index():
    """Return a simple HTML page for MJPEG video and separate audio streaming."""
    return """
    <html>
      <head>
        <title>Streaming Audio & Video</title>
      </head>
      <body>
        <h1>Streaming Audio & Video (MJPEG + separate audio)</h1>
        <img src="/video_feed" width="640" />
        <br>
        <audio controls autoplay>
          <source src="/audio_feed" type="audio/wav">
          Your browser does not support the audio element.
        </audio>
      </body>
    </html>
    """

def generate_video():
    """Generator that yields JPEG-encoded frames from the avatar’s queue."""
    while True:
        frame = global_avatar.encoded_frame_queue.get()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """MJPEG video streaming route."""
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_feed')
def audio_feed():
    """Audio streaming route: serve the audio file."""
    print("@ audio_feed, global_audio_path:", global_audio_path)
    return send_file(global_audio_path, mimetype='audio/wav')

# ----- WebRTC signaling using aiohttp and aiortc -----
# This part is used when --webrtc flag is given.
class AvatarVideoTrack(VideoStreamTrack):
    """
    A custom VideoStreamTrack that pulls raw frames from the avatar's raw_frame_queue.
    """
    def __init__(self, avatar, fps):
        super().__init__()
        self.avatar = avatar
        self.fps = fps

    async def recv(self):
        # Wait for a frame from the raw_frame_queue.
        frame = await asyncio.get_event_loop().run_in_executor(None, self.avatar.raw_frame_queue.get)
        if frame is None:
            raise asyncio.CancelledError()
        # Convert the raw BGR numpy frame to an av.VideoFrame.
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = None
        video_frame.time_base = 1 / self.fps
        return video_frame

async def offer(request):
    """
    HTTP POST endpoint that receives an SDP offer and returns an SDP answer.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    print("Created RTCPeerConnection for new client")

    # Add the custom video track from our avatar.
    video_track = AvatarVideoTrack(global_avatar, fps=args.fps)
    pc.addTrack(video_track)

    # Add an audio track using MediaPlayer to play the chosen audio file.
    player = MediaPlayer(global_audio_path)
    if player.audio:
        pc.addTrack(player.audio)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print("Generated SDP answer")
    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

# ----- Main entry point -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--skip_save_images", action="store_true",
                        help="Whether skip saving images for better generation speed calculation")
    # --stream activates the original Flask streaming mode.
    parser.add_argument("--stream", action="store_true",
                        help="Enable network streaming via a Flask web server")
    # --webrtc activates the WebRTC-based streaming mode.
    parser.add_argument("--webrtc", action="store_true",
                        help="Enable WebRTC streaming (video and audio together) via aiortc and aiohttp")
    args = parser.parse_args()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    if args.webrtc:
        # --- WebRTC streaming mode ---
        # For simplicity, use the first avatar and its first audio clip.
        for avatar_id in inference_config:
            data_preparation = inference_config[avatar_id]["preparation"]
            video_path = inference_config[avatar_id]["video_path"]
            bbox_shift = inference_config[avatar_id]["bbox_shift"]
            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=bbox_shift,
                batch_size=args.batch_size,
                preparation=data_preparation)
            audio_clips = inference_config[avatar_id]["audio_clips"]
            first_audio = next(iter(audio_clips.values()))
            print("Using audio for WebRTC streaming:", first_audio)
            global_avatar = avatar
            global_audio_path = first_audio
            break  # only one avatar for this example

        # Start inference in a background thread using the WebRTC method.
        inference_thread = threading.Thread(target=global_avatar.inference_webrtc, args=(global_audio_path, args.fps))
        inference_thread.start()

        # Set up and run the aiohttp signaling server.
        webrtc_app = web.Application()
        webrtc_app.router.add_post("/offer", offer)
        print("Starting WebRTC signaling server on http://0.0.0.0:5000/offer ...")
        web.run_app(webrtc_app, port=5000)
    elif args.stream:
        # --- Original Flask streaming mode (MJPEG + separate audio) ---
        for avatar_id in inference_config:
            data_preparation = inference_config[avatar_id]["preparation"]
            video_path = inference_config[avatar_id]["video_path"]
            bbox_shift = inference_config[avatar_id]["bbox_shift"]
            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=bbox_shift,
                batch_size=args.batch_size,
                preparation=data_preparation)
            audio_clips = inference_config[avatar_id]["audio_clips"]
            first_audio = next(iter(audio_clips.values()))
            print("Inferring using audio:", first_audio)
            global_avatar = avatar
            global_audio_path = first_audio
            # Run inference in a background thread.
            inference_thread = threading.Thread(target=avatar.inference, args=(first_audio, args.fps))
            inference_thread.start()
            break  # only one avatar for this example
        print("Starting Flask server for network streaming ...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        # --- Non-streaming mode: simply run inference and processing.
        for avatar_id in inference_config:
            data_preparation = inference_config[avatar_id]["preparation"]
            video_path = inference_config[avatar_id]["video_path"]
            bbox_shift = inference_config[avatar_id]["bbox_shift"]
            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=bbox_shift,
                batch_size=args.batch_size,
                preparation=data_preparation)
            audio_clips = inference_config[avatar_id]["audio_clips"]
            for audio_num, audio_path in audio_clips.items():
                print("Inferring using:", audio_path)
                avatar.inference(audio_path, args.fps)
