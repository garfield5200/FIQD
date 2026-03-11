# FIQD
The production process of the FIQD dataset, as well as the instructions for fine-tuning and testing based on this dataset.

Firstly, the dataset is provided in the link below:
链接: https://pan.baidu.com/s/13DEqt2ZRgH6PQqQeNrt3wg?pwd=fiqd 提取码: fiqd

The code for generating the dataset, including the FIQD_LAV-DF dataset, is located in the main branch.

The generation process of the dataset referred to some research:

https://kns.cnki.net/kcms2/article/abstract?v=BkbJkO_np9N5i9XSz2XR-uOoaZFgJczZrrcsDtG2uaTyKsSBrNey2UILDzqm0_YESFByVESaDXfxoZa_k9xPACePM5mZYE2d1WLBsNAx_z_Zn2HwMa23dR3sEoy4Y92oft8QZKGD-6rfpmKMf9u0t1aIG_BpwIP21RxQawar4fw=&uniplatform=NZKPT

https://github.com/EDVD-LLaMA/EDVD-LLaMA

https://github.com/zhipeixu/FakeShield

The fine-tuning of the large model refers to the following methods:

https://github.com/hiyouga/LLaMAFactory

https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory

We use CIDEr and CSS as evaluation metrics,The specific code is in the evaluate2.py file, which utilizes jieba for word segmentation and the bge-m3 text encoding model:

https://github.com/fxsjy/jieba

https://huggingface.co/BAAI/bge-m3
