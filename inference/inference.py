"""
TRUS 报告生成推理模块：单张/批量推理 + Gradio WebUI。
基于 EchoVLM 的推理流程精简适配，Qwen2.5-VL-7B + LoRA。
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_NAME, MODEL_LOCAL_PATH, OUTPUT_DIR, IMAGE_DIR,
    INFERENCE_CONFIG, TEST_JSON_PATH, PROCESSED_DIR,
)

# 系统提示
SYSTEM_PROMPT = "你是一名专注于经直肠超声（TRUS）的医学顾问，擅长前列腺超声影像分析与结构化报告生成。"

# 默认用户提示
DEFAULT_PROMPT = "请根据这张经直肠超声（TRUS）图像，生成一份结构化的前列腺超声诊断报告，包括前列腺尺寸、形态、包膜完整性、内部回声特征、CDFI血流信号及超声诊断印象。"


class TRUSReportGenerator:
    """TRUS 报告生成器，封装模型加载与推理。"""

    def __init__(self, lora_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.lora_path = lora_path
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model_path = str(MODEL_LOCAL_PATH) if MODEL_LOCAL_PATH else MODEL_NAME
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "device_map": self.device,
            "trust_remote_code": True,
        }

        if self.lora_path:
            from peft import PeftModel
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

        self.model.eval()

    def generate(self, image: Union[str, Path, Image.Image], prompt: Optional[str] = None) -> str:
        """单张图像推理，返回生成的报告文本。"""
        if prompt is None:
            prompt = DEFAULT_PROMPT

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image) if isinstance(image, (str, Path)) else image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **INFERENCE_CONFIG)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output[0]

    def generate_batch(self, images: List[Union[str, Path]], prompts: Optional[List[str]] = None) -> List[str]:
        """批量推理。"""
        results = []
        for i, img in enumerate(tqdm(images, desc="批量推理")):
            prompt = prompts[i] if prompts else None
            result = self.generate(img, prompt)
            results.append(result)
        return results


def run_single(image_path: str):
    """命令行单张推理。"""
    gen = TRUSReportGenerator()
    report = gen.generate(image_path)
    print("=" * 60)
    print("生成的 TRUS 报告:")
    print("-" * 60)
    print(report)
    print("=" * 60)
    return report


def run_batch(test_json: Optional[Path] = None, lora_path: Optional[str] = None):
    """从测试集 JSON 批量推理并保存预测。"""
    if test_json is None:
        test_json = TEST_JSON_PATH

    if not test_json.exists():
        print(f"测试集不存在: {test_json}")
        return

    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    gen = TRUSReportGenerator(lora_path=lora_path)

    predictions = []
    for item in tqdm(test_data, desc="推理"):
        img_rel = item["images"][0]
        img_path = PROCESSED_DIR / img_rel

        if not img_path.exists():
            print(f"[警告] 图像不存在: {img_path}")
            continue

        pred = gen.generate(str(img_path))
        predictions.append({
            "id": item["id"],
            "prediction": pred,
            "reference": item["messages"][-1]["content"],
            "metadata": item.get("metadata", {}),
        })

    save_path = OUTPUT_DIR / "test_predictions.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"预测结果已保存: {save_path}")
    print(f"共预测 {len(predictions)} 条")

    return predictions


def launch_webui():
    """启动 Gradio Web 演示界面。"""
    import gradio as gr

    gen = TRUSReportGenerator()

    def predict(image, prompt):
        if image is None:
            return "请上传一张 TRUS 超声图像。"
        if not prompt or not prompt.strip():
            prompt = DEFAULT_PROMPT
        return gen.generate(image, prompt)

    with gr.Blocks(title="TRUS 前列腺癌报告生成") as demo:
        gr.Markdown("# 🏥 TRUS 经直肠超声前列腺诊断报告生成")
        gr.Markdown("基于 Qwen2.5-VL-7B 的医学报告自动生成系统")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传 TRUS 超声图像")
                prompt_input = gr.Textbox(
                    label="提示词（可选）",
                    lines=2,
                    placeholder=DEFAULT_PROMPT,
                    value=DEFAULT_PROMPT,
                )
                submit_btn = gr.Button("生成报告", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="生成的诊断报告", lines=15, max_lines=30)

        submit_btn.click(
            fn=predict,
            inputs=[image_input, prompt_input],
            outputs=output_text,
        )

        gr.Examples(
            examples=[],
            inputs=[image_input, prompt_input],
        )

    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRUS 报告生成推理")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "batch", "webui"],
                        help="推理模式: single(单张), batch(批量), webui(Web界面)")
    parser.add_argument("--image", type=str, default=None, help="单张推理: 图像路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter 路径")
    parser.add_argument("--test_json", type=str, default=None, help="批量推理: 测试JSON路径")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.image:
            print("请用 --image 指定图像路径")
        else:
            run_single(args.image)
    elif args.mode == "batch":
        test_path = Path(args.test_json) if args.test_json else None
        run_batch(test_path, lora_path=args.lora_path)
    elif args.mode == "webui":
        launch_webui()
