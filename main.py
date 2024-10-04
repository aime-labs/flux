from pathlib import Path
import random
import argparse
import datetime
import base64
import io
import os
from PIL import Image

import torch
import numpy as np
from einops import rearrange

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5

from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "flux_dev"
DEFAULT_WORKER_AUTH_KEY = "2a14da16a70713bb3a4484b4ae5f681f"
VERSION = 1

class Inferencer():

    def __init__(self, device):
        self.device = torch.device(device)
        self.is_schnell = None
        self.model, self.ae, self.t5, self.clip = None, None, None, None


    def load_models(self, ckpt_path: str, name: str,  is_schnell: bool):
        t5_path = "google/t5-v1_1-xxl"
        clip_path = "openai/clip-vit-large-patch14"
        if ckpt_path:
            ckpt_path = os.path.join(ckpt_path, '')
            configs[name].ckpt_path = ckpt_path + "flux1-dev.safetensors"
            configs[name].ae_path = ckpt_path + "ae.safetensors"
            t5_path_ = ckpt_path + "models--google--t5-v1_1-xxl"
            if os.path.exists(t5_path_):
                t5_path = t5_path_
            clip_path_ = ckpt_path + "models--openai--clip-vit-large-patch14"
            if os.path.exists(clip_path_):
                clip_path = clip_path_
        self.is_schnell = is_schnell
        self.t5 = load_t5(self.device, t5_path, max_length=256 if is_schnell else 512)
        self.clip = load_clip(self.device, clip_path)
        self.model = load_flow_model(name, device=self.device)
        self.ae = load_ae(name, device=self.device)


    def decode_latents(self, x, height, width):
        # decode latents to pixel space
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        
        # bring into PIL format
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


    def gen_image(self, prompt, callback, width, height, num_steps, seed, guidance, progress_images=False, init_image=None, image2image_strength=0.8):
        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        print(f"generating {width}x{height} image with seed {opts.seed}")

        if init_image is not None:
            init_image = init_image.convert("RGB")
            init_image = np.array(init_image).astype(np.float32)
            init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 128.0
            init_image = init_image - 1.0; 
            init_image = init_image.unsqueeze(0) 
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (opts.height, opts.width))
            init_image = self.ae.encode(init_image.to())

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )
        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        progress_step = 1
        callback(None, progress_step, False, message='Preparing image...')

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        img = inp['img']
        img_ids = inp['img_ids']
        txt = inp['txt']
        txt_ids = inp['txt_ids']
        vec = inp['vec']

        # denoise initial noise
        #  inner loop of x = denoise(self.model, **inp, timesteps=timesteps, guidance=opts.guidance)

        progress_image = None
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = self.model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred

            if(progress_images) and (progress_step % 4 == 0):
                progress_image = self.decode_latents(img, opts.height, opts.width) 

            progress_step += 1
            callback(progress_image, progress_step, False, message='Denoising...')

        # decode latents to pixel space
        callback(progress_image, progress_step + 1, False, message='Decoding...')
        img = self.decode_latents(img, opts.height, opts.width)

        callback(img)


class ProcessOutputCallback():
    def __init__(self, api_worker, inferencer, model_name):
        self.api_worker = api_worker
        self.inferencer = inferencer
        self.model_name = model_name
        self.job_data = None

    def process_output(self, image, progress_step=100, finished=True, error=None, message=None):
        
        if error:
            print('error')
            self.api_worker.send_progress(100, None)
            image = Image.fromarray((np.random.rand(1024,1024,3) * 255).astype(np.uint8))
            return self.api_worker.send_job_results({'images': [image], 'error': error, 'model_name': self.model_name})
        else:
            if not finished:
                step_factor = self.job_data.get('image2image_strength') if self.job_data.get('image') else 1
                total_steps = int(self.job_data.get('steps') * step_factor) + 3
                progress_info = round((progress_step) * 100 / total_steps)

                if self.api_worker.progress_data_received:
                    progress_data = {'progress_message': message}
                    if image is not None:
                        progress_data['progress_images'] = [image]                       
                    return self.api_worker.send_progress(progress_info, progress_data)
            else:
                image_list = [image]
                self.api_worker.send_progress(100, None)
                return self.api_worker.send_job_results({'images': image_list, 'model_name': self.model_name})


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the AIME API server"
                        )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
                        )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help="Destination of model weigths"
                        )
    parser.add_argument(
        "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, 
        help="API server worker auth key",
    )
    return parser.parse_args()


def convert_base64_string_to_image(base64_string, width, height):
    if base64_string:
        base64_data = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_data)

        with io.BytesIO(image_data) as buffer:
            image = Image.open(buffer)
            return image.resize((width, height), Image.LANCZOS)


def set_seed(job_data):    
    seed = job_data.get('seed', -1)
    if seed == -1:
        random.seed(datetime.datetime.now().timestamp())
        seed = random.randint(1, 99999999)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    job_data['seed'] = seed
    return job_data


@torch.no_grad()
def main():
    args = load_flags()
    device = "cuda:" + str(args.gpu_id)
    torch.set_default_device(device)
    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name(), worker_version=VERSION)

    print("Loading models... ")
    inferencer = Inferencer(device)
    inferencer.load_models(args.ckpt_dir, "flux-dev", is_schnell=False)

    callback = ProcessOutputCallback(api_worker, inferencer, 'flux-dev')

    print("Waiting for jobs... ")

    while True:
        try:
            job_data = api_worker.job_request()
            print(f'Processing job {job_data.get("job_id")}...', end='', flush=True)
            job_data = set_seed(job_data)
            init_image = job_data.get('image')
            if init_image:
                init_image = convert_base64_string_to_image(
                    init_image,
                    job_data.get('width'), 
                    job_data.get('height')
                )
            callback.job_data = job_data           
            image = inferencer.gen_image(
                job_data.get('prompt'),
                callback.process_output,
#                job_data.get('num_samples', 1),
                job_data.get('width'), 
                job_data.get('height'), 
                job_data.get('steps'), 
                job_data.get('seed'),
                job_data.get('guidance'),
                job_data.get('provide_progress_images') == "decoded",
                init_image,
                job_data.get('image2image_strength')
            )
            print('Done')
        except ValueError as exc:
            print('Error')
            callback.process_output(None , None, True, f'{exc}\nChange parameters and try again')
            continue
        except torch.cuda.OutOfMemoryError as exc:
            print('Error - CUDA OOM')
            callback.process_output(None, None, True, f'{exc}\nReduce number image size and try again')
            continue
        except OSError as exc:
            print('Error')
            callback.process_output(None, None, True, f'{exc}\nInvalid image file')
            continue

if __name__ == "__main__":
    main()