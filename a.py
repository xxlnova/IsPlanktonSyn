
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

import diffusers
from diffusers.utils import load_image
from diffusers import AutoencoderKL, DDIMScheduler, ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection, AutoProcessor, Blip2ForConditionalGeneration,CLIPVisionModel
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig
from scipy.ndimage import gaussian_filter
from inversion import run as invert
from diffusers import StableDiffusionControlNetInpaintPipeline
from ip_adapter.pipeline_stable_diffusion_extra_cfg import StableDiffusionPipelineCFG
from ip_adapter.ip_adapter_instruct import IPAdapterInstruct
from src.frequency_utils import freq_exp
import json
def load_processed_records(records_path):
    """加载已处理的样本记录"""
    if os.path.exists(records_path):
        with open(records_path, 'r', encoding='utf-8') as f:
            return set(json.load(f))  # 用集合存储已处理的名称，方便快速查询
    return set()

def save_processed_record(records_path, sample_name):
    """保存已处理的样本名称到记录文件"""
    # 先加载现有记录
    processed = load_processed_records(records_path)
    # 添加新样本
    processed.add(sample_name)
    # 保存回文件
    with open(records_path, 'w', encoding='utf-8') as f:
        json.dump(list(processed), f, ensure_ascii=False, indent=2)

def clear_processed_records(records_path):
    """清空已处理记录（从头开始时使用）"""
    if os.path.exists(records_path):
        os.remove(records_path)


# ===================== 1. 核心工具函数（复用+适配着色任务） =====================
# def generate_caption(
#     image: Image.Image,
#     text: str = None,
#     decoding_method: str = "Nucleus sampling",
#     temperature: float = 1.0,
#     length_penalty: float = 1.0,
#     repetition_penalty: float = 1.5,
#     max_length: int = 50,
#     min_length: int = 1,
#     num_beams: int = 5,
#     top_p: float = 0.9,
# ) -> str:
#     """生成图像描述（复用Blip2，用于着色提示词增强）"""
#     if text is not None:
#         inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)
#         generated_ids = model.generate(** inputs)
#     else:
#         inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
#         generated_ids = model.generate(
#             pixel_values=inputs.pixel_values,
#             do_sample=decoding_method == "Nucleus sampling",
#             temperature=temperature,
#             length_penalty=length_penalty,
#             repetition_penalty=repetition_penalty,
#             max_length=max_length,
#             min_length=min_length,
#             num_beams=num_beams,
#             top_p=top_p,
#         )
#     result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return result
def generate_inpaint_mask(bw_img):
    # 简单版：将黑白图转为灰度，阈值分割生成遮罩（物体=白，背景=黑）
    gray = cv2.cvtColor(np.array(bw_img), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 阈值可调整
    # 确保mask是3通道（适配管道输入）
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mask)
def resize_img(input_image, max_side=512, min_side=512, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    """图像缩放（默认适配SD15的512分辨率）"""
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def get_depth_map(image, config):
    """生成深度图（用于保留黑白图的空间结构）"""
    depth_estimator = DPTForDepthEstimation.from_pretrained("./checkpoint/models/Intel").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("./checkpoint/models/Intel")
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(config.resolution, config.resolution),  # 适配512分辨率
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def get_canny_map(input_image_cv2):
    """生成Canny边缘图（用于保留黑白图的轮廓结构）"""

    # input_image_cv2 = cv2.Canny(input_image_cv2, 0, 1)
    input_image_cv2 = cv2.Canny(input_image_cv2, 50, 150)
    input_image_cv2 = input_image_cv2[:, :, None]
    input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
    kernel = np.ones((2, 2), np.uint8)
    input_image_cv2 = cv2.dilate(input_image_cv2, kernel, iterations=1)
    anyline_image = Image.fromarray(input_image_cv2)
    return anyline_image
def init_models(config):
    """初始化IP-Adapter和SD15管道（复用原有逻辑）"""
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    SD15_LOCAL_DIR = "/root/autodl-tmp/StyleSSP-main/local_sd15_model/"
    if config.choose_pipeline == "sd15":
        ip_ckpt = "./checkpoint/models/ip-adapter-instruct-sd15.bin"  # 指令型权重
        image_encoder_path = "/root/autodl-tmp/StyleSSP-main/local_clip_high/"
        pipe = StableDiffusionPipelineCFG.from_pretrained(
            SD15_LOCAL_DIR,
            scheduler=noise_scheduler,
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None,
            local_files_only=True
        )
        # print(f"SD15 UNet cross_attention_dim: {pipe.unet.config.cross_attention_dim}")
        ip_model = IPAdapterInstruct(
            sd_pipe=pipe,
            image_encoder_path=image_encoder_path,
            ip_ckpt=ip_ckpt,
            device=config.device,
            dtypein=torch.float16,
            num_tokens=16)
    return ip_model

# ===================== 2. 着色专用Dataset（适配黑白图+彩色参考图） =====================
class ColorizeDataset(Dataset):
    def __init__(self, bw_dir, ref_color_dir, resolution=512, use_single_ref=True, single_ref_idx=0):
        """
        着色数据集：加载黑白内容图和彩色参考图
        Args:
            bw_dir: 黑白内容图文件夹（输入）
            ref_color_dir: 彩色参考图文件夹（颜色风格来源）
            resolution: 统一分辨率（SD15用512）
            use_single_ref: 是否用单张参考图给所有黑白图着色
            single_ref_idx: 单参考图模式下的索引
        """
        self.resolution = resolution
        self.use_single_ref = use_single_ref

        # 加载黑白内容图（强制转为单通道黑白图）

        self.bw_paths = self._get_image_paths(bw_dir)
        if len(self.bw_paths) == 0:
            raise ValueError(f"黑白图文件夹 {bw_dir} 中无有效图片！")

        # 加载彩色参考图
        self.ref_paths = self._get_image_paths(ref_color_dir)
        if len(self.ref_paths) == 0:
            raise ValueError(f"彩色参考图文件夹 {ref_color_dir} 中无有效图片！")

        # 单参考图模式：固定一张参考图
        if self.use_single_ref:
            print('single_ref_idx',single_ref_idx)
            print('ref_paths',len(self.ref_paths))
            if single_ref_idx >= len(self.ref_paths):
                raise IndexError(f"参考图索引 {single_ref_idx} 超出范围（共 {len(self.ref_paths)} 张）")
            self.fixed_ref_path = self.ref_paths[single_ref_idx]
            self.fixed_ref_img = self._load_and_resize(self.fixed_ref_path, is_bw=False)

        # 多参考图模式：黑白图与参考图数量一致
        else:
            if len(self.bw_paths) != len(self.ref_paths):
                raise ValueError(f"多参考图模式需黑白图和参考图数量一致！当前黑白图 {len(self.bw_paths)} 张，参考图 {len(self.ref_paths)} 张")

    def _get_image_paths(self, dir_path):
        """获取文件夹中所有图片路径（筛选常见格式）"""
        image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        paths = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.lower().endswith(image_formats):
                paths.append(os.path.join(dir_path, filename))
        return paths

    def _load_and_resize(self, img_path, is_bw=True):
        """加载图像并缩放：黑白图强制转为单通道，参考图保留彩色"""
        try:
            img = Image.open(img_path)
            # 黑白图：转为单通道（L模式），再转回RGB（确保3通道输入）
            if is_bw:
                img = img.convert("RGB")  # L=单通道黑白，转回RGB便于模型输入
            else:
                img = img.convert("RGB")
            # 缩放到指定分辨率
            img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
            return img
        except Exception as e:
            raise RuntimeError(f"加载图像失败：{img_path}，错误：{e}")

    def __len__(self):
        return len(self.bw_paths)

    def __getitem__(self, idx):
        """返回：黑白图、彩色参考图、名称（用于保存）"""
        # 加载黑白内容图
        bw_path = self.bw_paths[idx]
        bw_img = self._load_and_resize(bw_path, is_bw=True)
        bw_name = os.path.splitext(os.path.basename(bw_path))[0]
        # 加载彩色参考图
        if self.use_single_ref:
            ref_img = self.fixed_ref_img
            ref_path = self.fixed_ref_path
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]
        else:
            ref_path = self.ref_paths[idx]
            ref_img = self._load_and_resize(ref_path, is_bw=False)
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]

        return {
            "bw_img": bw_img,       # 黑白内容图（3通道，像素值相同）
            "ref_img": ref_img,     # 彩色参考图（3通道）
            "bw_name": bw_name,     # 黑白图名称
            "ref_name": ref_name,
            "idx":idx # 参考图名称
        }

# ===================== 3. 主程序（着色核心逻辑） =====================
if __name__ == "__main__":
    # 1. 初始化配置
    if not os.path.exists("colorize_results"):
        os.makedirs("colorize_results")
    result_root = "colorize_results"
    records_path = os.path.join(result_root, "processed_records.json")

    model_type = Model_Type.SD15
    scheduler_type = Scheduler_Type.DDIM
    config = RunConfig(
        model_type=model_type,

        num_inference_steps=35,    # 512分辨率适配步数
        num_inversion_steps=35,
        num_renoise_steps=1,
        scheduler_type=scheduler_type,
        perform_noise_correction=False,
        seed=1234,
        resolution=512,            # SD15原生分辨率
        guidance_scale=7.5,        # SD15着色推荐引导强度
        style_guidance_scale=0.6,  # 增强颜色风格迁移强度
        content_guidance_scale=1.2, # 保留黑白图结构强度
        inv_guidance=0.1,
        # 批量文件夹路径

        # 新增：指定ControlNet类型（循环外加载一次）
    )
    device = config.device
    processed_samples = load_processed_records(records_path)
    print(f"已处理样本数量：{len(processed_samples)}")
    if len(processed_samples) > 0:
        print(f"已处理样本列表：{list(processed_samples)[:5]}...")  # 只打印前5个

    # 2. 加载Blip2（生成图像描述）
    # LOCAL_MODEL_DIR = "/root/autodl-tmp/StyleSSP-main/Blip/"
    # processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR)
    # model = Blip2ForConditionalGeneration.from_pretrained(
    #     LOCAL_MODEL_DIR, device_map="cuda", load_in_8bit=False, torch_dtype=torch.float16
    # ).eval()

    # 3. 初始化IP-Adapter（颜色特征提取）
    ip_instruct_model = init_models(config)

    # 4. 定义着色专用指令
    content_instruct_prompt = "preserve the structure and details of the image"
    color_instruct_prompt = "transfer the color palette and color style from the reference image"
    # content_instruct_prompt = "preserve the exact segmentation, curvature, and anatomical structure of the polychaete"
    # color_instruct_prompt = "transfer the color pattern"

    # 5. 创建Dataset和DataLoader
    dataset = ColorizeDataset(
        bw_dir=config.bw_dir,
        ref_color_dir=config.ref_color_dir,
        resolution=config.resolution,
        use_single_ref=config.use_single_ref,
        single_ref_idx=config.single_ref_idx
    )


    def custom_collate_fn(batch):
        # batch 是一个 list of dict，每个 dict 包含 PIL images 和 strings
        # 我们不合并图像，只提取第一个（因为 batch_size=1）
        return batch[0]  # 直接返回单个样本的 dict
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,  # Windows设为0
        drop_last=False,
        collate_fn=custom_collate_fn  # <-- 关键！
    )

    # ===================== 关键优化：ControlNet 循环外加载（仅加载一次） =====================
    controlnet = None
    controlnet_conditioning_scale = None
    # 根据config.control_type加载对应ControlNet（仅执行一次）
    if config.control_type == "canny":
        controlnet = ControlNetModel.from_pretrained(
            config.canny_controlnet_path_sd15,
            torch_dtype=torch.float16,
            use_safetensors=False,
            variant="fp16"
        ).to(device)
        controlnet_conditioning_scale = 0.8

    elif config.control_type == "depth":
        controlnet = ControlNetModel.from_pretrained(
            config.depth_controlnet_path_sd15,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        controlnet_conditioning_scale = 0.5

    elif config.control_type == "combine":
        # 组合Depth+Canny（预加载两个模型）
        controlnet = [
            ControlNetModel.from_pretrained(
                config.depth_controlnet_path_sd15,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device),
            ControlNetModel.from_pretrained(
                config.canny_controlnet_path_sd15,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)
        ]
        controlnet_conditioning_scale = [0.3, 0.8]

    elif config.control_type == "tile_canny":
        controlnet = [
            ControlNetModel.from_pretrained(
                config.tile_controlnet_path_sd15,
                torch_dtype=torch.float16,
            ).to(device),
            ControlNetModel.from_pretrained(
                config.canny_controlnet_path_sd15,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)
        ]
        controlnet_conditioning_scale = [0.25, 0.4]

    print(f"ControlNet 加载完成！类型：{config.control_type}")

    # 6. 预加载推理管道通用组件（循环外仅加载一次）
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "/root/autodl-tmp/StyleSSP-main/local_clip_high/",
        local_files_only=True,
        torch_dtype=config.dtype
    ).to(config.device)



    vae = AutoencoderKL.from_pretrained(
        "/root/autodl-tmp/StyleSSP-main/sd-vae-ft-mse/",
        torch_dtype=config.dtype,
        use_safetensors=True
    ).to(config.device)

    print("推理管道通用组件（VAE/ImageEncoder）加载完成！")

    # 7. 批量着色生成（循环内仅生成条件图和推理）
    total_samples = len(dataloader)
    skipped_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        # 提取批量数据

        bw_img = batch["bw_img"]  # 黑白内容图（PIL）
        ref_img = batch["ref_img"]  # 彩色参考图（PIL）
        bw_name = batch["bw_name"]
        ref_name = batch["ref_name"]
        sample_idx = batch["idx"]
        # print(bw_img)
        if bw_name in processed_samples:
            skipped_samples += 1
            print(f"\n跳过已处理样本 {batch_idx + 1}/{total_samples}：{bw_name}")
            continue

        print(f"\n处理第 {batch_idx+1}/{len(dataloader)} 个样本：")
        print(f"黑白图：{bw_name} | 颜色参考图：{ref_name}")


        # 8. 生成图像描述
        bw_image_prompt = (
             "microscopic image of Polychaeta larva or egg-bearing individual under bright-field illumination, "
            "elongated segmented body — NOT a complete adult worm, likely larval or small specimen, "
            "visible parapodia or setae — fine, hair-like projections along lateral or posterior margins, "
            "internal structure — visible clustered oocytes or yolk granules within body cavity, "
            "brightness gradient — peripheral edges brighter, central region darker, creating 'contour glow' effect, "
            "diffuse edges — fuzzy, blending into background, NOT sharp or defined, "
            "background pure black with scattered out-of-focus particles"

        )

        ref_image_prompt = (
             "Polychaeta larva or egg-bearing individual under phase-contrast microscopy, "
            "soft-bodied segmented organism — semi-transparent pale yellow or creamy white dominant color, "
            "with occasional orange-yellow pigment patches — possibly yolk or internal organs, "
            "edges — soft, slightly reflective, blending smoothly into background, NO hard borders, "
            "surface texture — gelatinous, mucous-like, or membranous, NOT smooth, NOT plastic, NOT crystalline, "
            "internal structure — visible clustered oocytes or empty spaces, NOT uniform, NOT opaque, "
            "peripheral edges appear brighter — creating 'contour glow' or 'halo' effect, NOT glowing, NOT neon, "
            "background pure black with soft focus biological particles, high contrast"

        )

        print(f"黑白图描述：{bw_image_prompt}")
        print(f"参考图颜色描述：{ref_image_prompt}")

        # 9. 提取特征嵌入
        color_embeddings = ip_instruct_model.get_decouple_embeds(
            pil_image=ref_img, prompt=ref_image_prompt, query=color_instruct_prompt
        )
        structure_embeddings = ip_instruct_model.get_decouple_embeds(
            pil_image=bw_img, prompt=bw_image_prompt, query=content_instruct_prompt
        )
        neg_color_embeddings = ip_instruct_model.get_decouple_embeds(
            pil_image=bw_img, prompt=bw_image_prompt, query="do not use this color"
        )
        neg_structure_embeddings = ip_instruct_model.get_decouple_embeds(
            pil_image=ref_img, prompt=ref_image_prompt, query="do not preserve this structure"
        )
        print(f"color_embeddings shape: {color_embeddings.shape}")
        print(f"structure_embeddings shape: {structure_embeddings.shape}")



        # 10. 生成ControlNet条件图（循环内仅生成图，不加载模型）
        cond_image = None
        if config.control_type == "canny":
            # 生成Canny边缘图（仅图像计算，无模型加载）
            bw_img_cv2 = np.array(bw_img)
            canny_img = get_canny_map(bw_img_cv2)
            cond_image = canny_img.resize((config.resolution, config.resolution))

        elif config.control_type == "depth":
            # 生成深度图（仅图像计算，无模型加载）
            depth_img = get_depth_map(bw_img, config)
            cond_image = depth_img.resize((config.resolution, config.resolution))

        elif config.control_type == "combine":
            # 生成Depth+Canny条件图
            depth_img = get_depth_map(bw_img, config)
            bw_img_cv2 = np.array(bw_img)
            canny_img = get_canny_map(bw_img_cv2)
            cond_image = [
                depth_img.resize((config.resolution, config.resolution)),
                canny_img.resize((config.resolution, config.resolution))
            ]

        elif config.control_type == "tile_canny":
            # 生成Tile+Canny条件图s
            tile_img = bw_img.resize((config.resolution, config.resolution))
            bw_img_cv2 = np.array(bw_img)
            canny_img = get_canny_map(bw_img_cv2)
            cond_image = [
                tile_img,
                canny_img.resize((config.resolution, config.resolution))
            ]

        # 11. 黑白图反演
        pipe_inversion, pipe_inference = get_pipes(
            model_type, scheduler_type, device=device, model_name="/root/autodl-tmp/StyleSSP-main/local_sd15_model/"
        )
        # print("pipe_inference type:", type(pipe_inversion))
        # print("Image encoder config:")
        # print(f"  hidden_size: {image_encoder.config.hidden_size}")
        # print(f"  projection_dim: {image_encoder.config.projection_dim}")
        # print(f"  model_type: {image_encoder.config.model_type}")
        _, inv_latent, _, all_latents = invert(
            bw_img,
            bw_image_prompt,
            config,
            pipe_inversion=pipe_inversion,
            pipe_inference=pipe_inference,
            do_reconstruction=False,
            feature_extractor=ip_instruct_model,
            style_embedding=color_embeddings,
            content_embedding=structure_embeddings,
            neg_style_embedding=neg_color_embeddings,
            neg_content_embedding=neg_structure_embeddings,
            enable_guidance=False,
            used_NPI_guidance=True
        )
        # print('inv_latent',inv_latent.shape)

        # 12. 频率域操作
        latent_h, latent_l, latent_sum = freq_exp(inv_latent, d_s=0.3, d_t=0.9, alpha=0.7, filter_type="gaussian_b")
        latent_l = latent_l.to(inv_latent.dtype)
        latent_h = latent_h.to(inv_latent.dtype)

        # 清理反演管道（仅反演用，推理用独立管道）
        del pipe_inversion, pipe_inference, all_latents
        torch.cuda.empty_cache()

        # 13. 加载推理管道（循环内加载）
        pipe_inference = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "/root/autodl-tmp/StyleSSP-main/local_sd15_model/",
            controlnet=controlnet,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

        pipe_inference.scheduler = UniPCMultistepScheduler.from_config(pipe_inference.scheduler.config)
        pipe_inference.unet.enable_gradient_checkpointing()

        # 关键修改1：加载IP-Adapter（老版本会自动嵌入到UNet的attn_processors，无需后续赋值）
        pipe_inference.load_ip_adapter(
            "./checkpoint/models/ipadapter/",
            subfolder='',
            weight_name="ip-adapter_sd15.safetensors",
            image_encoder_folder=None,
        )
        pipe_inference.set_ip_adapter_scale(0.9)




        # 14. 着色推理（后续代码不变，无需修改任何IP-Adapter相关属性）
        generator = torch.Generator(device="cpu").manual_seed(config.seed + batch_idx)
        save_name = f"{bw_name}"

        prompt = (
            "high-resolution microscopic image of Polychaeta larva or egg-bearing individual, "
            "ELONGATED SEGMENTED BODY — MUST be non-spherical, non-circular, NOT geometrically perfect, "
            "PARAPODIA OR SETAE — FINE, HAIR-LIKE PROJECTIONS ALONG LATERAL OR POSTERIOR MARGINS — NOT BOLD, NOT SOLID, MAY BE REFRACTIVE, "
            "INTERNAL STRUCTURE — VISIBLE CLUSTERED OOCYTES OR YOLK GRANULES — NOT UNIFORM, NOT GLOWING, "
            "DIFFUSE EDGES — FUZZY, GRADUAL TRANSITION TO BACKGROUND, NO HARD BORDERS, NO SHARP EDGES, NOT CUT OUT, "
            "TRANSPARENCY — HIGHLY SEMI-TRANSPARENT, LIGHT PASSES THROUGH, NOT OPAQUE, NOT GLOWING, "
            "COLOR — PALE YELLOW OR CREAMY WHITE DOMINANT, WITH OCCASIONAL ORANGE-YELLOW PIGMENT PATCHES AT INTERNAL ORGANS OR EGG CLUSTERS, "
            "BRIGHTNESS GRADIENT — PERIPHERAL EDGES SIGNIFICANTLY BRIGHTER THAN CENTRAL REGION — CREATING 'CONTOUR GLOW' EFFECT, NOT NEON, NOT OVEREXPOSED, "
            "TEXTURE — GELATINOUS, MUCOUS-LIKE, OR MEMBRANOUS, NOT SMOOTH, NOT PLASTIC, NOT CRYSTALLINE, NOT METALLIC, "
            "SURFACE — MAY SHOW MINOR TEXTURAL VARIATION (e.g., micro-folds or fine cilia), NOT FLAT, NOT POLISHED, "
            "BACKGROUND — PURE BLACK (RGB 0,0,0) with sparse out-of-focus biological particles, "
            "HIGH CONTRAST BETWEEN STRUCTURE AND BACKGROUND, "
            "MUST STRICTLY FOLLOW INPUT GRAYSCALE STRUCTURE — DO NOT ADD/REMOVE PARAPODIA, SETAE, OR OOCYTE CLUSTERS, "
            "scientific, photorealistic, no artifacts, avoid cartoonish colors, neon glow, plastic look, uniform fill, hard edges"

        )

        negative_prompt = (
            "grayscale, black and white, oversaturated colors, neon glow, metallic sheen, "
            "cartoon, anime, painting, sketch, watercolor, thick borders, bold stripes, "
            "pixelated, artificial patterns, halos, sparkles, 3D render, plastic look, "
            "noise, artifacts, text, watermark, low res, blurry, "
            "complete organism, whole animal, head, tail, eyes, legs, fins, "
            "sharp edges, hard borders, cut-out shape, geometric shape, circular, spherical, "
            "uniform color, solid fill, flat color, gradient fill, "
            "crystalline, glassy, smooth, polished, reflective surface, "
            "glowing particle, glowing edges, glowing interior, "
            "fibers too long, filaments too thick, brush strokes, "
            "multiple distinct objects, companion particle, "
            "gray background, white background, no particles, "
            "biological features not matching polychaetes — e.g., no cell membrane, no nucleus, no flagella, "
            "perfect symmetry, identical left/right sides, "
            "too many textures, too little texture, "
            "texture inside — only surface if any, "
            "overly bright, overexposed, washed out, "
            "red/blue/green/purple dominant color, "
            "yellow/orange too saturated, "
            "edge enhancement, sharpened edges, "
            "vector line, clipart, logo, icon, "
            "branched, segmented, knotted, coiled, twisted, multi-tailed, symmetrical"

        )

        output = pipe_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            eta=1.0,
            mask_image=generate_inpaint_mask(bw_img),
            image=bw_img,
            control_image=cond_image,
            ip_adapter_image=ref_img,
            generator=generator,
            latents=latent_l,
            guidance_scale=config.guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            npi_interp=0.5,
            style_embeddings_instruct=color_embeddings,
            content_embeddings_instruct=structure_embeddings,
            style_guidance_scale=config.style_guidance_scale,
            content_guidance_scale=config.content_guidance_scale,
            ip_instruct_model=ip_instruct_model,
            CSD_model=None,
            inv_guidance=config.inv_guidance,
            feature_extractor=ip_instruct_model,
            do_NPI=False,
        ).images[0]

        # 15. 保存结果
        output_path = os.path.join(config.result_path, f"colorized_{save_name}.png")
        # output_path_bw = os.path.join(config.result_path, f"{save_name}.png")
        output.save(output_path)
        # bw_img.save(output_path_bw)
        print(f"着色结果保存至：{output_path}\n" + "="*50)
        # ===================== 核心：保存已处理记录 =====================
        save_processed_record(records_path, bw_name)
        print(f"已记录样本 {bw_name} 到断点文件")
        # 清理推理管道（避免显存累积）
        del pipe_inference, output, latent_l, latent_h
        torch.cuda.empty_cache()

    # 循环结束后，清理预加载的大模型
    del controlnet, vae, image_encoder, ip_instruct_model
    torch.cuda.empty_cache()

    print("\n批量着色完成！所有结果已保存至：", config.result_path)