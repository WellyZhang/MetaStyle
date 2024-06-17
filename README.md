# MetaStyle

This repo contains the PyTorch code for our AAAI 2019 paper.  

[MetaStyle: Three-Way Trade-Off Among Speed, Flexibility, and Quality in Neural Style Transfer](http://wellyzhang.github.io/attach/aaai19zhang.pdf)  
Chi Zhang, Yixin Zhu, Song-Chun Zhu  
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2019.  

An unprecedented booming has been witnessed in the research area of artistic style transfer ever since Gatys et al. introduced the neural method. One of the remaining challenges is to balance a trade-off among three critical aspects - speed, flexibility, and quality: (i) the vanilla optimization-based algorithm produces impressive results for arbitrary styles, but is unsatisfyingly slow due to its iterative nature, (ii) the fast approximation methods based on feed-forward neural networks generate satisfactory artistic effects but bound to only a limited number of styles, and (iii) feature-matching methods like AdaIN achieve arbitrary style transfer in a real-time manner but at a cost of the compromised quality. We find it considerably difficult to balance the trade-off well merely using a single feed-forward step and ask, instead, whether there exists an algorithm that could adapt quickly to any style, while the adapted model maintains high efficiency and good image quality. Motivated by this idea, we propose a novel method, coined MetaStyle, which formulates the neural style transfer as a bilevel optimization problem and combines learning with only a few post-processing update steps to adapt to a fast approximation model with satisfying artistic effects, comparable to the optimization-based methods for an arbitrary style. The qualitative and quantitative analysis in the experiments demonstrates that the proposed approach achieves high-quality arbitrary artistic style transfer effectively, with a good trade-off among speed, flexibility, and quality.

![framework](./images/readme/procedure.png)

# Examples

We show some samples below. The left column shows the content image and its style-free representation. The rest of the figure displays different stylized content images.

![examples](./images/readme/prologue.png)

The following figure shows comparison with other methods.

![comparison](./images/readme/compare.png)

A video demo could be found [here](https://vimeo.com/303954291).

# Dependencies

**Important**
* PyTorch (>= 0.4.0)
* CUDA and cuDNN

See ```requirements.txt``` for a full list of packages required.

# Usage

## Training

To train a model of your own:

1. First download a content image dataset and a style image dataset. In this paper, we use [MS-COCO](http://cocodataset.org/#download) and [WikiArt](https://www.kaggle.com/c/painter-by-numbers). 
2. Run
```
python src/main.py train --content-dataset <path-to-your-content-dataset> --style-dataset <path-to-your-style-dataset> --cuda 1
```

Usually, the default parameters should work. However, you are always welcome to fine tune yourself. The training process could be monitored using Tensorboard. Since bilevel optimization requires "second-order" gradient computation, the training process might take a long time depending on the GPU you have, and the GPU memory consumption is huge. 

We provide our pre-trained model [here](https://pan.baidu.com/s/1kb6gvnkWa3ivY_1gYzy-dQ?pwd=jytg).

## Fast Training

To adapt the model to a new style, run
```
python src/main.py fast --content-dataset <path-to-your-content-dataset> --style-image <path-to-your-style-image> --model <path-to-your-trained-model> --cuda 1
```

Usually fast adaptation requires only 100 to 200 post update steps and could be done in less than 30 seconds, depending on the GPU you have.

## Testing

To stylize a content image, run
```
python src/main.py test --content-image <path-to-your-content-image> --output-image <path-to-your-output-image> --model <path-to-your-trained-model> --cuda 1
```

# Citation

If you find the paper and the code helpful, please cite us.
```
@inproceedings{zhang2019metastyle,
    title={MetaStyle: Three-Way Trade-Off Among Speed, Flexibility, and Quality in Neural Style Transfer},
    author={Zhang, Chi and Zhu, Yixin and Zhu, Song-Chun},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
    year={2019}
}
```

# Acknowledgement

This project is impossible to finish without the help of my colleagues and the following open-source implementations. 

* [Fast Neural Style](https://github.com/jcjohnson/fast-neural-style)
* [PyTorch Examples](https://github.com/pytorch/examples/tree/master/fast_neural_style)
* [Original MAML](https://github.com/cbfinn/maml)
* [PyTorch MAML](https://github.com/katerakelly/pytorch-maml)
* [OpenAI Reptile](https://blog.openai.com/reptile/)
* [AdaIN](https://github.com/xunhuang1995/AdaIN-style)
* [PyTorch AdaIN](https://github.com/naoto0804/pytorch-AdaIN)

# License

MetaStyle is freely available for non-commercial use, and may be redistributed under these conditions. Please see the [license](./LICENSE) for further details. For commercial license, please contact the authors.
