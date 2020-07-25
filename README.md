## [RefinedBox: Refining for Fewer and High-quality Object Proposals](https://yun-liu.github.io/papers/(Neurocomputing'2020)RefinedBox%20-%20Refining%20for%20Fewer%20and%20High-quality%20Object%20Proposals.pdf)

### Introduction

<img src="https://raw.githubusercontent.com/yun-liu/RefinedBox/master/framework.png" width="2427">

Recently, object proposal generation has shown value for various vision tasks, such as object detection, semantic instance segmentation, multi-label image classification, and weakly supervised learning, by hypothesizing object locations. We are motivated by the fact that many traditional proposal methods generate dense proposals to cover as many objects as possible but that i) they usually fail to rank these proposals properly and ii) the number of proposals is very large. For example, the well-known object proposal generation methods, Edge Boxes and Selective Search, can achieve high detection recall with thousands of proposals per image. But the large number of generated proposals makes subsequent analyses difficult due to the large number of false alarms and heavy computation load. o significantly reduce the number of proposals, we design a computationally ightweight neural network to refine the initial object proposals. The refinement consists of two parallel processes, re-ranking and box regression. The proposed network can share convolutional features with other high-level tasks by joint training, so the proposal refinement can be very fast. We show a joint training example of object detection in this paper. Extensive experiments demonstrate that our method can achieve state-of-the-art performance with a few proposals compared with some well-known proposal generation methods.

### License

RefinedBox is released under the MIT License (refer to the LICENSE file for details).

### Citations

If you find RefinedBox useful in your research, please consider citing:

	@article{liu2020refinedbox,
	  title={RefinedBox: Refining for Fewer and High-quality Object Proposals},
	  author={Liu, Yun and Li, Shi-Jie and Cheng, Ming-Ming},
	  journal={Neurocomputing},
	  volume={406},
	  pages={106--116},
	  year={2020},
	  publisher={Elsevier}
	}

### Acknowledgment

This code is based on Faster R-CNN. Thanks to the contributors of Faster R-CNN.

	@inproceedings{ren2015faster,
	  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
	  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
	  booktitle={Advances in neural information processing systems},
	  pages={91--99},
	  year={2015}
	}
