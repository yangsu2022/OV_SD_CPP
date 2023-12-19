import gc
import warnings
from pathlib import Path
from diffusers import DiffusionPipeline
import torch
import openvino as ov

from typing import Union, Optional, Any, List, Dict
from transformers import CLIPTokenizer, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.image_processor import VaeImageProcessor

warnings.filterwarnings("ignore")

TEXT_ENCODER_OV_PATH = Path("../models/lcm/dreamshaper_v7/FP16_dyn/text_encoder/openvino_model.xml")
UNET_OV_PATH = Path("../models/lcm/dreamshaper_v7/FP16_dyn/unet/openvino_model.xml")
VAE_DECODER_OV_PATH = Path("../models/lcm/dreamshaper_v7/FP16_dyn/vae_decoder/openvino_model.xml")


def load_orginal_pytorch_pipeline_componets(skip_models=False, skip_safety_checker=False):
    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    feature_extractor = pipe.feature_extractor if not skip_safety_checker else None
    safety_checker = pipe.safety_checker if not skip_safety_checker else None
    text_encoder, unet, vae = None, None, None
    if not skip_models:
        text_encoder = pipe.text_encoder
        text_encoder.eval()
        unet = pipe.unet
        unet.eval()
        vae = pipe.vae
        vae.eval()
    del pipe
    gc.collect()
    return (
        scheduler,
        tokenizer,
        feature_extractor,
        safety_checker,
        text_encoder,
        unet,
        vae,
    )


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder mode.
    Function accepts text encoder model, and prepares example inputs for conversion,
    Parameters:
        text_encoder (torch.nn.Module): text_encoder model from Stable Diffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    input_ids = torch.ones((1, 77), dtype=torch.long)
    # switch model to inference mode
    text_encoder.eval()

    # disable gradients calculation for reducing memory consumption
    with torch.no_grad():
        # Export model to IR format
        ov_model = ov.convert_model(
            text_encoder,
            example_input=input_ids,
            input=[
                (-1, 77),
            ],
        )
    # ov.save_model(ov_model, ir_path)
    ov.save_model(ov_model, ir_path,compress_to_fp16=False)
        
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"Text Encoder successfully converted to IR and saved to {ir_path}")

def convert_unet(unet: torch.nn.Module, ir_path: Path):
    """
    Convert U-net model to IR format.
    Function accepts unet model, prepares example inputs for conversion,
    Parameters:
        unet (StableDiffusionPipeline): unet from Stable Diffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    # prepare inputs
    dummy_inputs = {
        "sample": torch.randn((1, 4, 64, 64)),
        "timestep": torch.ones([1]).to(torch.float32),
        "encoder_hidden_states": torch.randn((1, 77, 768)),
        "timestep_cond": torch.randn((1, 256)),
    }
    unet.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=dummy_inputs)
    # ov.save_model(ov_model, ir_path)
    ov.save_model(ov_model, ir_path,compress_to_fp16=False)
        
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"Unet successfully converted to IR and saved to {ir_path}")

def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model for decoding to IR format.
    Function accepts vae model, creates wrapper class for export only necessary for inference part,
    prepares example inputs for conversion,
    Parameters:
        vae (torch.nn.Module): VAE model frm StableDiffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    vae_decoder = VAEDecoderWrapper(vae)
    latents = torch.zeros((1, 4, 64, 64))

    vae_decoder.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(vae_decoder, example_input=latents)
    # ov.save_model(ov_model, ir_path)
    ov.save_model(ov_model, ir_path,compress_to_fp16=False)
        
    del ov_model
    cleanup_torchscript_cache()
    print(f"VAE decoder successfully converted to IR and saved to {ir_path}")

class OVLatentConsistencyModelPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae_decoder: ov.Model,
        text_encoder: ov.Model,
        tokenizer: CLIPTokenizer,
        unet: ov.Model,
        scheduler: None,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.register_to_config(unet=unet)
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor
        self.vae_scale_factor = 2**3
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        if prompt_embeds is None:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(text_input_ids, share_inputs=True, share_outputs=True)
            prompt_embeds = torch.from_numpy(prompt_embeds[0])

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil"
                )
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt"
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = torch.randn(shape, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 4,
        lcm_origin_steps: int = 50,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # do_classifier_free_guidance = guidance_scale > 0.0
        # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            prompt_embeds=prompt_embeds,
        )

        with open("prompt_embeds.txt", "w") as file:
            for row in prompt_embeds.view(-1):  
                file.write(str(row.item()) + " ")

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, original_inference_steps=lcm_origin_steps)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latent variable
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            latents,
        )
        # print("latents: ", latents)
        print("latents.shape: ", latents.shape)

        # with open("latents.txt", "w") as file:
        #     for row in latents.view(-1):  
        #         file.write(str(row.item()) + " ")

        bs = batch_size * num_images_per_prompt

        # 5. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = self.get_w_embedding(w, embedding_dim=256)
        # print("w: ", w)
        print("w_embedding.shape: ", w_embedding.shape)

        # print(w_embedding)

        # 6. LCM MultiStep Sampling Loop:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                ts = torch.full((bs,), t, dtype=torch.long)

                # model prediction (v-prediction, eps, x)
                model_pred = self.unet([latents, ts, prompt_embeds, w_embedding], share_inputs=True, share_outputs=True)[0]
                # compute the previous noisy sample x_t -> x_t-1
                print("model_pred: ", model_pred.flatten()[0:5])

                latents, denoised = self.scheduler.step(
                    torch.from_numpy(model_pred), t, latents, return_dict=False
                )
                print("latents after step: ",latents.flatten()[0:5])

                progress_bar.update()

        if not output_type == "latent":
            image = torch.from_numpy(self.vae_decoder(denoised / 0.18215, share_inputs=True, share_outputs=True)[0])
            image, has_nsfw_concept = self.run_safety_checker(
                image, prompt_embeds.dtype
            )
        else:
            image = denoised
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

skip_conversion = (
    TEXT_ENCODER_OV_PATH.exists()
    and UNET_OV_PATH.exists()
    and VAE_DECODER_OV_PATH.exists()
)

(
    scheduler,
    tokenizer,
    feature_extractor,
    safety_checker,
    text_encoder,
    unet,
    vae,
) = load_orginal_pytorch_pipeline_componets(skip_conversion)

if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder(text_encoder, TEXT_ENCODER_OV_PATH)
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

del text_encoder
gc.collect()

if not UNET_OV_PATH.exists():
    convert_unet(unet, UNET_OV_PATH)
else:
    print(f"Unet will be loaded from {UNET_OV_PATH}")
del unet
gc.collect()

if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder(vae, VAE_DECODER_OV_PATH)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

del vae
gc.collect()