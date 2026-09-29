import json
import re
from dataclasses import dataclass
from PIL import Image
import os
import pymupdf
from tqdm import tqdm
from datetime import datetime
import glob
import base64
import html

@dataclass
class Content:
    image: Image
    image_path: str
    txt: str
    
class BaseDataset():
    def __init__(self, config):
        self.config = config
        self.IM_FILE = lambda doc_name,index: f"{self.config.extract_path}/{doc_name}_{index}.png"
        self.TEXT_FILE = lambda doc_name,index: f"{self.config.extract_path}/{doc_name}_{index}.txt"
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub("\\.pdf$", "", sample["doc_id"]).split("/")[-1] 
        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")
    
    def load_data(self, use_retreival=True):
        path = self.config.sample_path
        if use_retreival:
            try:
                assert(os.path.exists(self.config.sample_with_retrieval_path))
                path = self.config.sample_with_retrieval_path
            except:
                print("Use original sample path!")
                
        assert(os.path.exists(path))
        with open(path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            
        return samples
    
    def dump_data(self, samples, use_retreival=True):
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent = 4)
        
        return path
    
    def load_latest_results(self):
        print(self.config.result_dir)
        path = find_latest_json(self.config.result_dir)
        with open(path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples, path
    
    def dump_reults(self, samples):
        os.makedirs(self.config.result_dir, exist_ok=True)
        path = os.path.join(self.config.result_dir, self.time + ".json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent = 4)
        return path

    def dump_html(self, samples, ans_key="ans"):
        os.makedirs(self.config.result_dir, exist_ok=True)
        html_path = os.path.join(self.config.result_dir, self.time + ".html")

        def img_to_base64(img_path):
            if not img_path or not os.path.exists(img_path):
                return None
            try:
                with open(img_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

        def guess_mime(img_path):
            ext = os.path.splitext(img_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg", ".gif": "image/gif",
                        ".bmp": "image/bmp", ".webp": "image/webp"}
            return mime_map.get(ext, "image/png")

        parts = []
        parts.append("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"UTF-8\">")
        parts.append(f"<title>MHier-Agent 推理报告 - {html.escape(self.time)}</title>")
        parts.append("<style>")
        parts.append("""
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px 20px; background: #f5f5f5; color: #222; }
h1 { text-align: center; font-size: 1.6em; margin-bottom: 8px; color: #1a1a1a; }
.subtitle { text-align: center; color: #666; margin-bottom: 32px; font-size: 0.9em; }
.card { background: #fff; border-radius: 10px; padding: 28px 24px; margin-bottom: 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.card h2 { font-size: 1.15em; border-bottom: 2px solid #1a73e8; padding-bottom: 8px;
           margin-bottom: 16px; color: #1a73e8; }
.label { font-weight: 600; color: #333; margin-bottom: 6px; font-size: 0.95em; }
.answer-box { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 16px 18px;
              border-radius: 4px; margin: 8px 0 16px; line-height: 1.8; white-space: pre-wrap; }
.question-box { background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px;
                border-radius: 4px; margin: 8px 0 16px; line-height: 1.7; }
.agent-response { background: #f8f9fa; border-left: 4px solid #9e9e9e; padding: 12px 16px;
                  border-radius: 4px; margin: 8px 0; line-height: 1.7; white-space: pre-wrap;
                  font-size: 0.9em; max-height: 400px; overflow-y: auto; }
summary { cursor: pointer; font-weight: 600; color: #555; padding: 6px 0;
          user-select: none; font-size: 0.95em; }
summary:hover { color: #1a73e8; }
details { margin-bottom: 12px; }
.gallery { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0; }
.gallery img { max-width: 48%; max-height: 480px; border: 1px solid #ddd;
               border-radius: 6px; object-fit: contain; background: #fafafa; }
.text-block { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
              padding: 12px 16px; margin: 6px 0; font-size: 0.88em; line-height: 1.65;
              white-space: pre-wrap; max-height: 260px; overflow-y: auto; }
.doc-id { color: #888; font-size: 0.85em; }
.footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 32px;
          padding-top: 16px; border-top: 1px solid #e0e0e0; }
""")
        parts.append("</style></head><body>")
        parts.append(f"<h1>MHier-Agent 推理报告</h1>")
        parts.append(f"<p class=\"subtitle\">生成时间: {html.escape(self.time)}</p>")

        for i, sample in enumerate(samples, 1):
            question = sample.get(self.config.question_key, "(无问题)")
            final_ans = sample.get(ans_key, "(未生成)")
            # handle dict answer from sum agent
            if isinstance(final_ans, dict):
                final_ans = final_ans.get("Answer", str(final_ans))
            doc_id = sample.get("doc_id", "")
            images = sample.get("images", [])
            texts = sample.get("texts", [])
            agent_details = sample.get(ans_key + "_agent_details", {})
            messages = sample.get(ans_key + "_message", "")

            parts.append(f"<div class=\"card\">")
            parts.append(f"<h2>#{i} 样本 <span class=\"doc-id\">({html.escape(str(doc_id))})</span></h2>")

            parts.append(f"<p class=\"label\">问题</p>")
            parts.append(f"<div class=\"question-box\">{html.escape(str(question))}</div>")

            parts.append(f"<p class=\"label\">最终答案</p>")
            parts.append(f"<div class=\"answer-box\">{html.escape(str(final_ans))}</div>")

            if images:
                parts.append(f"<details><summary>检索到的图片 ({len(images)} 张)</summary>")
                parts.append(f"<div class=\"gallery\">")
                for img_path in images:
                    b64 = img_to_base64(img_path)
                    if b64:
                        mime = guess_mime(img_path)
                        parts.append(f"<img src=\"data:{mime};base64,{b64}\" "
                                     f"alt=\"{html.escape(os.path.basename(img_path))}\" "
                                     f"title=\"{html.escape(os.path.basename(img_path))}\">")
                    else:
                        parts.append(f"<p style=\"color:#999;\">图片不可用: {html.escape(img_path)}</p>")
                parts.append(f"</div></details>")

            if texts:
                parts.append(f"<details><summary>检索到的文本片段 ({len(texts)} 条)</summary>")
                for j, txt in enumerate(texts, 1):
                    parts.append(f"<p class=\"label\" style=\"margin-top:8px;\">#{j}</p>")
                    parts.append(f"<div class=\"text-block\">{html.escape(str(txt))}</div>")
                parts.append(f"</details>")

            if agent_details:
                parts.append(f"<details open><summary>各智能体回复</summary>")
                for agent_name, agent_resp in agent_details.items():
                    parts.append(f"<p class=\"label\" style=\"margin-top:10px;\">{html.escape(str(agent_name))}</p>")
                    parts.append(f"<div class=\"agent-response\">{html.escape(str(agent_resp))}</div>")
                parts.append(f"</details>")
            elif messages:
                parts.append(f"<details><summary>所有智能体回复</summary>")
                parts.append(f"<div class=\"agent-response\">{html.escape(str(messages))}</div>")
                parts.append(f"</details>")

            parts.append(f"</div>")

        parts.append("<div class=\"footer\">Generated by MHier-Agent &mdash; Powered by Multi-Agent Document QA System</div>")
        parts.append("</body></html>")

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(parts))

        return html_path
    
    def load_retrieval_data(self):
        assert(os.path.exists(self.config.sample_with_retrieval_path))
        with open(self.config.sample_with_retrieval_path, 'r') as f:
            samples = json.load(f)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_retrieval_data(sample)
        return samples
    
    def load_sample_retrieval_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []
        # If retrieval already stored concrete texts/images, return them directly
        direct_texts = sample.get("texts")
        direct_images = sample.get("images")
        if self._is_str_list(direct_texts) or self._is_str_list(direct_images):
            if self._is_str_list(direct_texts):
                texts = direct_texts[:self.config.top_k]
            if self._is_str_list(direct_images):
                images = direct_images[:self.config.top_k]
            return question, texts, images
        if self.config.use_mix:
            if self.config.r_mix_key in sample:
                for page in sample[self.config.r_mix_key][:self.config.top_k]:
                    if page in sample[self.config.r_image_key]:
                        origin_image_path = ""
                        origin_image_path = content_list[page].image_path
                        images.append(origin_image_path)
                    if page in sample[self.config.r_text_key]:
                        texts.append(content_list[page].txt.replace("\n", ""))
        else:
            if self.config.r_text_key in sample:
                if self._is_str_list(sample[self.config.r_text_key]):
                    texts = sample[self.config.r_text_key][:self.config.top_k]
                else:
                    for page in sample[self.config.r_text_key][:self.config.top_k]:
                        texts.append(content_list[page].txt.replace("\n", ""))
            if self.config.r_image_key in sample:
                if self._is_str_list(sample[self.config.r_image_key]):
                    images = sample[self.config.r_image_key][:self.config.top_k]
                else:
                    for page in sample[self.config.r_image_key][:self.config.top_k]:
                        origin_image_path = ""
                        origin_image_path = content_list[page].image_path
                        images.append(origin_image_path)
                        
        return question, texts, images

    def _is_str_list(self, value):
        if not isinstance(value, list):
            return False
        return all(isinstance(item, str) for item in value)
    
    def load_full_data(self):
        samples = self.load_data(use_retreival=False)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_full_data(sample)
        return samples
    
    def load_sample_full_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []
        
        if self.config.page_id_key in sample:
            sample_no_list = sample[self.config.page_id_key]
        else:
            sample_no_list = [i for i in range(0,min(len(content_list),self.config.vlm_max_page))]
        for page in sample_no_list:
            texts.append(content_list[page].txt.replace("\n", ""))
            origin_image_path = ""
            origin_image_path = content_list[page].image_path
            images.append(origin_image_path)
                        
        return question, texts, images
      
    def load_processed_content(self, sample: dict, disable_load_image=True)->list[Content]:
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        content_list = []
        for page_idx in range(self.config.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(im_file)
            txt = self.load_txt(text_file)
            content_list.append(Content(image=img, image_path=im_file, txt=txt)) 
        return content_list
    
    def load_image(self, file):
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        max_length = self.config.max_character_per_page
        with open(file, 'r', encoding='utf-8') as file:
            content = file.read()
        content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return content[:max_length]
    
    def extract_content(self, resolution=144):
        samples = self.load_data()
        for sample in tqdm(samples):
            self._extract_content(sample, resolution=resolution)
            
    def _extract_content(self, sample, resolution=144):
        max_pages=self.config.max_page
        os.makedirs(self.config.extract_path, exist_ok=True)
        image_list = list()
        text_list = list()
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        with pymupdf.open(os.path.join(self.config.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:max_pages]):
                # save page as an image
                im_file = self.IM_FILE(doc_name,index)
                if not os.path.exists(im_file):
                    im = page.get_pixmap(dpi=resolution)
                    im.save(im_file)
                image_list.append(im_file)
                # save page text
                txt_file = self.TEXT_FILE(doc_name,index)
                if not os.path.exists(txt_file):
                    text = page.get_text("text")
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                text_list.append(txt_file)
                
        return image_list, text_list
    
def extract_time(file_path):
    file_name = os.path.basename(file_path)
    time_str = file_name.split(".json")[0]
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M")

def find_latest_json(result_dir):
    pattern = os.path.join(result_dir, "*-*-*-*-*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith('_results.json')]
    if not files:
        print(f"Json file not found at {result_dir}")
        return None
    latest_file = max(files, key=extract_time)
    return latest_file
