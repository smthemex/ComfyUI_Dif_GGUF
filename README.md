# ComfyUI_Dif_GGUF
Easy quant comfyUI origin models to gguf , and esay use it, save more disk...

# Tips
* fix high Vram error，修复高内存占用的bug
* add gguf clip support
* find it in model/loader,replace the original unet_loader to gguf_loader
* 节点在模型/加载目录下,对于大部分单unet加载的,替换原始unet加载器为gguf加载器即可,hidream-O1 暂时未支持 

1.Installation  
----

In the ./ComfyUI/custom_nodes directory, run the following:   

```
git clone https://github.com/smthemex/ComfyUI_Dif_GGUF
```

2.requirements  
----
If you have gguf install,don't need to install requirements.txt

```
pip install -r requirements.txt

```

3.checkpoints 
----
Any diffusion gguf,for example :    
[klein9b](https://huggingface.co/wikeeyang/Flux2-Klein-9B-True-V3)  
[comfy-org-gguf](https://huggingface.co/smthem/comfy-org-gguf)

```
├── ComfyUI/models/
|     ├── diffusion_models/
|        ├──Flux2-Klein-9B-True-V3-Q8_0.gguf # optional
|     ├── gguf/
|        ├──Flux2-Klein-9B-True-V3-Q8_0.gguf # optional

```

4.Example
----
![boogu](https://github.com/smthemex/ComfyUI_Dif_GGUF/blob/main/example_workflows/example.png)
![boogu](https://github.com/smthemex/ComfyUI_Dif_GGUF/blob/main/example_workflows/example-boogu.png)
![klein](https://github.com/smthemex/ComfyUI_Dif_GGUF/blob/main/example_workflows/example-klein.png)
![krea2](https://github.com/smthemex/ComfyUI_Dif_GGUF/blob/main/example_workflows/example-krea2.png)


