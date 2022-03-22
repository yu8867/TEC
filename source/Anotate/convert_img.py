# -*- coding: utf-8 -*-
# %%
from tqdm import tqdm
from pathlib import Path
from glob import glob
from PIL import Image
from IPython.display import display
import torch as th
import torch.nn as nn
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer
import numpy as np
import random
# fix seed
seed = 0
np.random.seed(seed)
random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True

# %%
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# %%
# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
# use 100 diffusion steps for fast sampling
options['timestep_respacing'] = '100'
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))


# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
# use 27 diffusion steps for very fast sampling
options_up['timestep_respacing'] = 'fast27'
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel()
      for x in model_up.parameters()))


# Create CLIP model.
clip_model = create_clip_model(device=device)
clip_model.image_encoder.load_state_dict(
    load_checkpoint('clip/image-enc', device))
clip_model.text_encoder.load_state_dict(
    load_checkpoint('clip/text-enc', device))


def ndarray_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
#     display(Image.fromarray(reshaped.numpy()))
    return Image.fromarray(reshaped.numpy())

# %%


def convert(text):
    prompt = text
    batch_size = 1
    guidance_scale = 3.0
    upsample_temp = 0.997

    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    model_kwargs = dict(
        tokens=th.tensor([tokens] * batch_size, device=device),
        mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
    )

    cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)

    model.del_cache()
    samples = diffusion.p_sample_loop(
        model,
        (batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    model.del_cache()
    print("image")
#     show_images(samples)



    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    model_kwargs = dict(
        low_res=((samples+1)*127.5).round()/127.5 - 1,
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    model_up.del_cache()
    up_shape = (batch_size, 3,
                options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()
    print("up_imgae")
    return ndarray_images(up_samples)


# %%

# 絶対パス
txts = glob('/workspace/data/outputs/*.txt')
print(txts)
p_inputs = Path('/workspace/data/inputs')
p_img = Path('/workspace/data/img')

picture_label = []

for txt in txts:
    p_txt = Path(txt)

    # file_nameにはその作品のファイル名が入る 例:'/workspace/outputs/モノグラム.txt'の場合'モノグラム.txt'
    file_name = p_txt.name
    print(file_name)

    # 翻訳した作品を改行区切りでリストに追加
    with p_txt.open(mode='r') as f:
        input_txt = f.read().split('\n')

    # 翻訳前(原文)の作品を改行区切りでリストに追加
    with (p_inputs/file_name).open(mode='r') as f:
        ja_txt = f.read().split('\n')
    
    # title_nameにはその作品のタイトル名が入る
    # 例:'/workspace/outputs/モノグラム.txt'の場合'モノグラム'
    title_name = p_txt.stem

    # /workspace/img/title_nameのフォルダ作成。
    # parentsで上のフォルダも再帰的に作成、exist_okで既にフォルダが存在してもエラー回避
    img_folder = (p_img/title_name)

    if img_folder.exists():
        continue
    img_folder.mkdir(parents=True, exist_ok=True)

    for i, x in enumerate(input_txt):
        # 日本語(原文)を表示(例外処理をしているのは、日本語の方のテキストファイルの最後の1行に不要な改行が入っていることがあるためです)
        try:
            print('Japanse :', ja_txt[i])
        except IndexError:
            continue
        # 英語の文章を表示
        print('English :', x)

        # convert関数からの戻り値の画像を[0]~[最後の行-1].png形式で保存
        pil_img = convert(x)
        # 表示
        display(pil_img)
        # 保存
        pil_img.save(img_folder/fr'{i}.png')

        # picture_label.append(int(input("0:not picture 1:ok")))

# %%
