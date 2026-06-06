from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # =================================================================================
        # 🔥🔥🔥 优化开关：检测是否需要运行 SAM 分支 🔥🔥🔥
        # 如果 masks_list 为 None，说明是纯文本/多模态指令微调，不需要分割
        # =================================================================================
        run_sam_branch = (masks_list is not None)
        if run_sam_branch and images is None:
            raise ValueError("masks_list is provided but images (SAM input) is None!")
        
        # 1. SAM Image Encoder (最吃显存的部分)
        # 只有在需要分割时才计算，否则直接跳过，节省大量显存
        image_embeddings = None
        if run_sam_branch:
            image_embeddings = self.get_visual_embs(images)
            batch_size = image_embeddings.shape[0]
            assert batch_size == len(offset) - 1
        
        # 2. 准备 SEG token mask (辅助逻辑)
        # 虽然这部分计算量不大，但如果不需要 SAM，也没必要跑
        seg_token_mask = None
        if run_sam_branch:
            # 利用 labels != -100，屏蔽掉所有属于 prompt(query) 里的 [SEG]
            valid_mask = labels[:, 1:] != -100
            seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx) & valid_mask
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            # hack for IMAGE_TOKEN_INDEX
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask], #消融feature情况下是255,Ours是511
                dim=1,
            )
        # =================================================================================
        # 3. LLM Forward (核心部分：NPR Projector 和 LLM 都在这里训练)
        # =================================================================================
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
        # =================================================================================
        # 4. 早期退出 (Early Exit)
        # 如果没有 mask，直接返回文本 Loss，不再往下跑 SAM 解码器
        # =================================================================================
        if not run_sam_branch:
            # 确保返回的字典包含所有必要的 key，防止训练循环报错，但值为 0
            return {
                "loss": output.loss, # 这里的 loss 只是 CE Loss (Text)
                "ce_loss": output.loss,
                "mask_bce_loss": torch.tensor(0.0).to(output.loss.device),
                "mask_dice_loss": torch.tensor(0.0).to(output.loss.device),
                "mask_loss": torch.tensor(0.0).to(output.loss.device),
            }
        # =================================================================================
        # 5. 原有的 SAM 分割逻辑 (只有 run_sam_branch=True 才会执行)
        # =================================================================================
        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        if last_hidden_state.dim() == 2 and seg_token_mask.dim() == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            if pred_embeddings[i].shape[0] == 0:
                h, w = resize_list[i]
                # 创建一个 [0, H, W] 的空 Tensor
                empty_mask = torch.zeros((0, h, w), device=image_embeddings.device, dtype=image_embeddings.dtype)
                pred_masks.append(empty_mask)
                continue

            # 2. 如果是假图 (Fake)，正常跑 SAM 流程
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # ✅ 修正点2：这里也直接用 resize_list[i] 作为 original_size
            # resize_list[i] 本身就是 tuple (H, W)，完美符合 interpolate 的要求
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],      # input_size 和 original_size 用同一个没问题，因为我们没有对原图做额外 crop
                original_size=resize_list[i],   # 👈 关键修改：不再依赖 label_list
            )
            pred_masks.append(pred_mask[:, 0])
               
        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            gt_mask = gt_mask.to(dtype=pred_mask.dtype)
            
            if pred_mask.shape[0] > gt_mask.shape[0]:
                pred_mask = pred_mask[:gt_mask.shape[0]]
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            if gt_mask.shape[0] > 0:
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "pred_masks": pred_masks,
            "logits": output,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            if isinstance(outputs.hidden_states, tuple):
                all_hidden_states = []
                for step_states in outputs.hidden_states:
                    # 如果返回的是 tuple (包含每一层的输出)，取最后一层
                    if isinstance(step_states, (tuple, list)):
                        h = step_states[-1]
                    # 如果返回的已经是 tensor (直接是最后一层的输出)，直接使用
                    else:
                        h = step_states
                    
                    # 防御性补丁：如果被意外切成了 2 维 (seq_len, hidden_dim)
                    # 强行给它补回第 0 维的 batch_size: (1, seq_len, hidden_dim)
                    if h.dim() == 2:
                        h = h.unsqueeze(0)
                        
                    all_hidden_states.append(h)
                # 沿着序列长度维度 (dim=1) 完美拼接
                output_hidden_states = torch.cat(all_hidden_states, dim=1)
            else:
                output_hidden_states = outputs.hidden_states[-1]
            # =========================================================
            
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            prompt_len = input_ids.shape[1]
            seg_token_mask[:, :prompt_len - 1] = False
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255), dtype=torch.bool, device=seg_token_mask.device), #消融情况下是255,Ours是511
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=seg_token_offset.device), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks

    def evaluate_analyse(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        # 创建可视化输出目录
        output_dir = "./vis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # =========================================================
        # 第一阶段：无梯度常规生成 (获取基础输出)
        # =========================================================
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
                output_hidden_states=True,
                output_scores=True,    
                return_dict_in_generate=True,
            )
            
            output_ids = outputs.sequences

            # 提取最后一层隐藏状态用于 SAM 掩码生成
            if isinstance(outputs.hidden_states, tuple):
                all_hidden_states = []
                for step_states in outputs.hidden_states:
                    h = step_states[-1] if isinstance(step_states, (tuple, list)) else step_states
                    if h.dim() == 2:
                        h = h.unsqueeze(0)
                    all_hidden_states.append(h)
                output_hidden_states = torch.cat(all_hidden_states, dim=1)
            else:
                output_hidden_states = outputs.hidden_states[-1]

        
        try:
            # 打印每一步生成的 Token 和它的概率
            print("\n--- 文本生成置信度分析 ---")
            for step_idx, step_logits in enumerate(outputs.scores):
                # step_logits shape: (batch_size, vocab_size)
                probs = torch.softmax(step_logits[0], dim=-1) # 转为概率分布
                top_prob, top_idx = torch.max(probs, dim=-1)
                gen_token = tokenizer.decode([top_idx])
                # 重点观察模型输出 'fake' 或者 'real' 时的 top_prob 有多高
                print(f"Token: '{gen_token:<10}' | 概率: {top_prob.item():.4f}")
            
            # =========================================================
            # 第二阶段：【终极分析】梯度显著性归因 (多目标追踪：定性词 + SEG)
            # =========================================================
            print("\n🚀 开始执行【终极归因分析：梯度显著性 (Gradient Saliency)】...")
            output_ids_single = output_ids[0]
            prompt_len = input_ids.shape[1]
            
            # 1. 动态寻找我们需要追踪的目标 Token
            targets_to_analyze = []
            
            # 寻找定性词：'fake' 或 'real'
            for pos in range(prompt_len, len(output_ids_single)):
                tid = output_ids_single[pos].item()
                token_str = tokenizer.decode([tid]).strip().lower()
                if "fake" in token_str or "real" in token_str:
                    targets_to_analyze.append((token_str, pos))
                    break # 找到第一个核心判定词即可
                    
            # 寻找掩码标记：[SEG]
            seg_positions = (output_ids_single == self.seg_token_idx).nonzero(as_tuple=True)[0]
            if len(seg_positions) > 0:
                targets_to_analyze.append(("[SEG]", seg_positions[0].item()))

            if not targets_to_analyze:
                print("⚠️ 未在生成序列中找到 'fake'/'real' 或 '[SEG]' token，跳过归因分析。")

            # 2. 开始逐个追踪！
            for target_name, target_pos in targets_to_analyze:
                print(f"\n🔍 正在追踪目标 Token: '{target_name}' (序列位置: {target_pos}) 的核心因果来源...")
                
                # 截取该目标 Token 之前的所有上下文
                context_ids = output_ids[:, :target_pos]
                # 这是模型当时预测出的那个真实的词 ID
                target_token_id = output_ids_single[target_pos].item()
                
                # [基础对齐逻辑]
                vocab_size = len(tokenizer)
                aligned_tokens = []
                for tid_tensor in context_ids[0]:
                    tid = tid_tensor.item()
                    if tid == -200: 
                        aligned_tokens.extend([f"[IMG_PATCH_{p_idx}]" for p_idx in range(256)])
                    elif tid < 0 or tid >= vocab_size:
                        aligned_tokens.append(f"[UNK_{tid}]")
                    else:
                        token_str = tokenizer.convert_ids_to_tokens(tid)
                        if isinstance(token_str, bytes):
                            token_str = token_str.decode('utf-8', errors='ignore')
                        if token_str and token_str.startswith(' '):
                            token_str = token_str.replace(' ', '')
                        aligned_tokens.append(str(token_str))

                # [终极自适应修复与梯度反向传播]
                with torch.enable_grad():
                    prep_func = getattr(self, 'prepare_inputs_labels_for_multimodal', getattr(self.get_model(), 'prepare_inputs_labels_for_multimodal', None))
                    
                    if prep_func is not None:
                        import inspect
                        sig = inspect.signature(prep_func)
                        kwargs = {'input_ids': context_ids, 'images': images_clip}
                        if 'past_key_values' in sig.parameters: kwargs['past_key_values'] = None
                        if 'labels' in sig.parameters: kwargs['labels'] = None
                        if 'attention_mask' in sig.parameters: kwargs['attention_mask'] = None
                        if 'position_ids' in sig.parameters: kwargs['position_ids'] = None
                        
                        prep_res = prep_func(**kwargs)
                        
                        inputs_embeds = None
                        if isinstance(prep_res, dict):
                            inputs_embeds = prep_res.get('inputs_embeds')
                        elif isinstance(prep_res, (list, tuple)):
                            for item in prep_res:
                                if isinstance(item, torch.Tensor) and item.dim() == 3 and item.is_floating_point():
                                    inputs_embeds = item
                                    break
                    else:
                        inputs_embeds = self.model.get_input_embeddings()(context_ids)

                    if inputs_embeds is None:
                        raise ValueError("无法提取 inputs_embeds！")

                    # 开启梯度追踪
                    inputs_embeds.requires_grad_()
                    inputs_embeds.retain_grad()

                    # 前向传播
                    outputs_grad = self.model(
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=False,
                        return_dict=True
                    )

                    # 提取对应目标的 Logit 并反向传播
                    last_hidden = outputs_grad[0] 
                    next_token_logits = self.lm_head(last_hidden[0, -1, :])
                    
                    # ⚠️ 关键修改：针对当前正在追踪的目标 Token 求导！
                    target_logit = next_token_logits[target_token_id]

                    self.model.zero_grad()
                    if hasattr(self, 'lm_head'):
                        self.lm_head.zero_grad()
                    target_logit.backward()

                    # 计算 Saliency
                    h_grad = inputs_embeds.grad
                    saliency_scores = (h_grad[0] * inputs_embeds[0]).norm(dim=-1).float().detach().cpu().numpy()
                
                # [截断对齐与归一化]
                valid_len = min(len(aligned_tokens), len(saliency_scores))
                valid_tokens = aligned_tokens[:valid_len]
                valid_saliency = saliency_scores[:valid_len]
                valid_saliency = valid_saliency / (valid_saliency.sum() + 1e-9)

                # 计算宏观区域聚合贡献度
                # 1. 图像特征区域 (Index: 36 ~ 291)
                img_start_idx, img_end_idx = 36, 291
                if valid_len > img_end_idx:
                    # Python切片是左闭右开，所以结束索引要 +1
                    image_total_contribution = valid_saliency[img_start_idx : img_end_idx + 1].sum()
                else:
                    # 动态兜底：如果长度不够，则根据字符串包含 "IMG_PATCH" 动态求和
                    image_total_contribution = sum([score for i, score in enumerate(valid_saliency) if "IMG_PATCH" in valid_tokens[i]])

                # 2. 先验提示词区域 (Index: 329 ~ 357)
                hint_start_idx, hint_end_idx = 329, 357
                if valid_len > hint_end_idx:
                    hint_total_contribution = valid_saliency[hint_start_idx : hint_end_idx + 1].sum()
                else:
                    hint_total_contribution = 0.0

                # [保存文本报告]
                safe_name = target_name.replace(" ", "").replace("[", "").replace("]", "")
                file_saliency = os.path.join(output_dir, f"saliency_for_{safe_name}_top15.txt")
                
                with open(file_saliency, "w", encoding="utf-8") as f:
                    f.write("==================================================\n")
                    f.write(f"🏆 目标: '{target_name}' | 因果贡献度 (Gradient Saliency) Top 15\n")
                    f.write("==================================================\n")
                    
                    top_indices = valid_saliency.argsort()[::-1][:15]
                    for rank, idx in enumerate(top_indices):
                        token_str = valid_tokens[idx]
                        score = valid_saliency[idx]
                        f.write(f"Top {rank+1:<2} | Index: {idx:<4} | 真实贡献度: {score*100:.2f}% | Token: '{token_str}'\n")
                        print(f"  --> Top {rank+1:<2} | 贡献: {score*100:.2f}% | Token: '{token_str}'")

                    f.write("\n==================================================\n")
                    f.write(f"🧩 宏观区域聚合贡献度 (Block-wise Aggregation)\n")
                    f.write("==================================================\n")
                    f.write(f"▶ 视觉证据区总贡献 (Image Patches, Index {img_start_idx}-{img_end_idx}): {image_total_contribution*100:.2f}%\n")
                    f.write(f"▶ 先验提示词总贡献 (Forensic Hint, Index {hint_start_idx}-{hint_end_idx}): {hint_total_contribution*100:.2f}%\n")

                    
                    f.write("\n==================================================\n")
                    f.write(f"✅ 完整序列核对 (标记贡献度 > 1% 的特征)\n")
                    f.write("==================================================\n")
                    for idx in range(valid_len):
                        token_str = valid_tokens[idx]
                        score = valid_saliency[idx]
                        mark = f" ⭐⭐⭐ [因果核心!]" if score > 0.01 else ""
                        f.write(f"Index: {idx:<4} | 真实贡献: {score*100:.2f}% | Token: '{token_str:<15}' {mark}\n")
                        
                # [绘制曲线图]
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(15, 4))
                    # 为不同的目标设置不同的颜色: fake/real 用橙色, [SEG] 用绿色
                    line_color = '#f97316' if target_name != '[SEG]' else '#10b981'
                    
                    plt.plot(valid_saliency, color=line_color, linewidth=1.5)
                    plt.fill_between(range(len(valid_saliency)), valid_saliency, color=line_color, alpha=0.2)
                    plt.title(f"Target: '{target_name}' - Causal Saliency Map", fontsize=14, fontweight='bold')
                    plt.xlabel("Aligned Token Index", fontsize=12)
                    plt.ylabel("Contribution", fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    
                    max_idx = valid_saliency.argmax()
                    max_val = valid_saliency.max()
                    max_token_str = valid_tokens[max_idx]
                    
                    plt.axvline(x=max_idx, color='#dc2626', linestyle='--', alpha=0.8)
                    plt.text(max_idx + 2, max_val, f"Max: '{max_token_str}'\nScore: {max_val*100:.1f}%", 
                             color='#dc2626', verticalalignment='top', fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#dc2626', boxstyle='round,pad=0.3'))
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"saliency_plot_{safe_name}.png"), dpi=300)
                    plt.close()
                except ImportError:
                    pass

        except Exception as e:
            import traceback
            print(f"梯度分析时出错: {e}")
            traceback.print_exc()

        # =========================================================
        # 第三阶段：原始分割解码器逻辑 (生成 Mask)
        # =========================================================
        with torch.no_grad():
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            prompt_len = input_ids.shape[1]
            seg_token_mask[:, :prompt_len - 1] = False
            
            # hack for IMAGE_TOKEN_INDEX
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255), dtype=torch.bool, device=seg_token_mask.device), 
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=seg_token_offset.device), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks