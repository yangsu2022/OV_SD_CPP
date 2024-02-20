# https://hf-mirror.com/deinferno/taesd-openvino

from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVStableDiffusionPipeline
from optimum.intel.openvino.modeling_diffusion import OVModelVaeDecoder, OVModelVaeEncoder, OVBaseModel
from diffusers.training_utils import set_seed
from diffusers import LMSDiscreteScheduler

# Create class wrappers which allow us to specify model_dir of TAESD instead of original pipeline dir
# text2image only need decoder

class CustomOVModelVaeDecoder(OVModelVaeDecoder):
    def __init__(
        self, model, parent_model, ov_config = None, model_dir = None,
    ):
        super(OVModelVaeDecoder, self).__init__(model, parent_model, ov_config, "vae_decoder", model_dir)
        
class CustomOVModelVaeEncoder(OVModelVaeEncoder):
    def __init__(
        self, model, parent_model, ov_config = None, model_dir = None,
    ):
        super(OVModelVaeEncoder, self).__init__(model, parent_model, ov_config, "vae_encoder", model_dir)

# To align with C++ pipeline, set seed, scheduler and prompt
lmsscheduler = LMSDiscreteScheduler.from_pretrained("OpenVINO/stable-diffusion-1-5-fp32", subfolder="scheduler")

ovsd_pipe = OVStableDiffusionPipeline.from_pretrained("OpenVINO/stable-diffusion-1-5-fp32", compile=False, scheduler=lmsscheduler)

ovsd_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
ovsd_pipe.compile()

set_seed(42)

prompt = "plant pokemon in jungle" 
# prompt = "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"

ovsd_output = ovsd_pipe(prompt, num_inference_steps=20, output_type="pil")

ovsd_output.images[0].save("result_ovsd.png")

print("Inject TAESD")

taesd_pipe = OVStableDiffusionPipeline.from_pretrained("OpenVINO/stable-diffusion-1-5-fp32", compile=False, scheduler=lmsscheduler)

# save taesd model
taesd_dir = snapshot_download(repo_id="deinferno/taesd-openvino", local_dir="../models/sd/taesd-openvino")
print(f"{taesd_dir}/vae_decoder/openvino_model.xml")

taesd_pipe.vae_decoder = CustomOVModelVaeDecoder(model = OVBaseModel.load_model(f"{taesd_dir}/vae_decoder/openvino_model.xml"), parent_model = taesd_pipe, model_dir = taesd_dir)
taesd_pipe.vae_encoder = CustomOVModelVaeEncoder(model = OVBaseModel.load_model(f"{taesd_dir}/vae_encoder/openvino_model.xml"), parent_model = taesd_pipe, model_dir = taesd_dir)

taesd_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
taesd_pipe.compile()

set_seed(42)

taesd_output = taesd_pipe(prompt, num_inference_steps=20, output_type="pil")
taesd_output.images[0].save("result_taesd.png")
