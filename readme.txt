机器：222.29.51.165 (A40)
运行环境：conda activate fcil




FineFMPL论文代码：
即本文件夹下的代码：具体的方法为：

使用-文本prompt(1个和1个（视觉特征映射（用glo、obj特征），对象特征分别由原始注意力提取、1xfg注意力提取+cls)）；2分支，对象权重（各自设置），对象分支（做MHSA前的scale映射（含sqrt操作）、做最后的聚合）；clip_logit和cache_logit温度0.01和0.03，权重为1、各自设置；每阶段都是全分类


训练命令：
bash train_cub200.sh
bash train_cifar100.sh
bash train_mini_imagenet.sh

测试命令（inference）：
bash test_cub200.sh
bash test_cifar100.sh
bash test_mini_imagenet.sh






说明如下：
1. 这个代码缓解了NAN（自己新加的神经网络出现NAN的情况下，会重新用float计算后转为half）


2. 这个代码是3个数据集共用的代码，直接运行相应的bash文件即可。唯一不同的是3个数据集的超参（共2个超参）不同。
    
    if dataset_name == 'cub200':
        factor = 0.5 #对象分支的权重
    elif dataset_name == 'cifar100':
        factor = 2.0 #对象分支的权重
    elif dataset_name == 'mini_imagenet':
        factor = 0.5 #对象分支的权重

    factor_clip = 1 #tip_logits = factor_clip * clip_logits + factor_cache * cache_logits

    if dataset_name == 'cub200':
        factor_cache = 1.5
    elif dataset_name == 'cifar100':
        factor_cache = 1.0
    elif dataset_name == 'mini_imagenet':
        factor_cache = 0.5
    

3. CIFAR数据集，当取factor=1, factor_cache=1时，就是会出现NAN，非常奇怪。但确实可能就是超参的问题。目前的原因分析下来是：半精度网络的梯度计算超出half范围
解决方案：

“序号0：版本10使用纯全精度网络；BS64；对象权重0，cache_logit权重1”  会导致CUB上性能降低。这个可以彻底解决NAN问题

“序号1：版本10+跨模态映射先float32操作再转回float16“ 可以避免在CIFAR上出现NAN,但性能会下降。这个版本可以帮助跑VIT-L模型，但CUB上batch只能开到32，影响最终性能。






其他说明：


文件夹：“避免出现NAN的代码”，采用了梯度剪切策略避免半精度网络的梯度爆炸问题，但性能会下降。梯度爆炸的现象只在特定的超参下会出现，目前只在CIFAR数据集上，对象权重为1，cache_logits权重为1时出现。因此，目前的策略是不采用这个超参，同时也不用这个梯度剪切。

缓解这个问题的另一个办法就是减小学习率

但上述文件夹需要尝试很多超参数，比如梯度裁剪的限度值等，才可能有效






文件夹：“序号0：版本10使用纯全精度网络；BS64；对象权重0，cache_logit权重1”

这个就是版本10的代码用全精度的网络和新加模块，性能会降低些，速度会变慢。但应该不会再出现NAN的问题。
目前时间紧迫，先不用这个代码。





文件夹：“序号1：版本10+跨模态映射先float32操作再转回float16“

可以避免在CIFAR上出现NAN,但性能会下降
这个版本可以帮助跑VIT-L模型，但CUB上batch只能开到32，影响最终性能。





文件夹：“使用-文本prompt(1个和1个（视觉特征映射（用glo、obj特征），对象特征分别由原始注意力提取、1xfg注意力提取+cls)）；2分支，对象权重（各自设置），对象分支（做MHSA前的scale映射（含sqrt操作）、做最后的聚合）；clip_logit和cache_logit温度0.01和0.03，权重为1、各自设置；每阶段都是全分类”