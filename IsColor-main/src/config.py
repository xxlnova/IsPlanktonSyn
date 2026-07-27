
from dataclasses import dataclass
import torch

from src.eunms import Model_Type, Scheduler_Type

@dataclass
class RunConfig:
    model_type : Model_Type = Model_Type.SD15
    scheduler_type : Scheduler_Type = Scheduler_Type.DDIM
    seed: int = 7865
    num_inference_steps: int = 4
    num_inversion_steps: int = 4
    inv_guidance_scale: float = 1.5
    guidance_scale: float =5.0#
    style_guidance_scale: float = 1.2#1.0
    content_guidance_scale: float = 1.0#0.5
    inv_style_guidance_scale: float = 0.0 # 5000
    inv_content_guidance_scale: float = 0.0
    inv_neg_style_guidance_scale: float = 0.0 #
    inv_neg_content_guidance_scale: float = 0.0
    get_grad_guidance: bool = True
    inv_guidance: float = 0.9
    num_renoise_steps: int = 9
    max_num_renoise_steps_first_step: int = 5
    inversion_max_step: float = 1.0
    device = 'cuda'# if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    useIP = True
    resolution: int = 512

    # Average Parameters
    average_latent_estimations: bool = True
    average_first_step_range: tuple = (0, 5)
    average_step_range: tuple = (8, 10)

    # Noise Regularization
    noise_regularization_lambda_ac: float = 20.0
    noise_regularization_lambda_kl: float = 0.065
    noise_regularization_num_reg_steps: int = 4
    noise_regularization_num_ac_rolls: int = 5  

    # Noise Correction
    perform_noise_correction: bool = True

    # model path
    # tile_controlnet_path = "./checkpoint/controlnet-tile-sdxl-1.0"
    # canny_controlnet_path = "./checkpoint/MistoLine"
    # depth_controlnet_path = "diffusers/controlnet-depth-sdxl-1.0-small"

    canny_controlnet_path_sd15 = '/root/autodl-tmp/StyleSSP-main/checkpoint/models/canny/'
    depth_controlnet_path_sd15 = '/root/autodl-tmp/StyleSSP-main/checkpoint/models/depth/'
    tile_controlnet_path_sd15 = '/root/autodl-tmp/StyleSSP-main/checkpoint/models/tile/'

    control_type = "tile_canny" # "tile" or "canny" or "depth" or "combine" or "tile_canny", where combine means use depth and canny
    choose_pipeline = 'sd15'#'sd15'

    # image param
    result_path = "./results2/001Aphanizomenon_flosaquae"
    result_root = "/root/autodl-tmp/StyleSSP-main/duandian"
    style_image_dir = "path/to/your/style/image/file" # such as ./data/style/7.jpg
    content_image_dir = "path/to/your/style/image/content/file" # such as ./data/content/7.jpg
    style_image_prompt = None # the prompt for style image, if None, use blip2 to generate
    resolution: int = 512

    # bw_dir = "/root/autodl-tmp/StyleSSP-main/dataset/DAPlankton/DAPlankton/DAPlankton_sea/CS/Chroococcales"  # 黑白图文件夹
    # ref_color_dir = "/root/autodl-tmp/StyleSSP-main/dataset/DAPlankton/DAPlankton/DAPlankton_sea/ref/Chroococcales"  # 彩色参考图文件夹
    bw_dir = "/root/autodl-tmp/StyleSSP-main/dataset/DAPlankton/DAPlankton/DAPlankton_sea/IFCB/Aphanizomenon_flosaquae"  # 黑白图文件夹
    ref_color_dir = "/root/autodl-tmp/StyleSSP-main/dataset/DAPlankton/DAPlankton/DAPlankton_sea/ref/Aphanizomenon_flosaquae"  # 彩色参考图文件夹
    use_single_ref = True,  # 单参考图模式（一张彩色图给所有黑白图着色）
    single_ref_idx = 0  # 选择参考图文件夹中第1张图
    def __post_init__(self):
        pass