# Gaussian VAE

**Yola25**  
**March 2026**

> 注：为了数学推导的方便，这里不太区分原书中表达概率分布的不同记号 $p$ 和 $q$。在下文推导中，近似后验统一写作 $q_\theta(z\mid x)$，生成模型相关分布写作 $p_\phi$。

## Notation

- $p(z)$：先验分布；
- $q_\theta(z\mid x)$：编码器，近似后验；
- $p_\phi(x\mid z)$：解码器，生成模型；
- $p_\phi(z\mid x)$：真实后验。

## Figure 1: VAE

```text
x --encoder θ--> z --decoder φ--> x'
```

VAE 是一个隐变量生成模型（latent variable generative model）。它假设观测数据 $x$ 并不是直接生成的，而是先在隐空间采样一个隐变量 $z$，再由 $z$ 生成 $x$。因此，VAE 的生成过程可以写为

$$
z \sim p(z), \qquad x \sim p_\phi(x\mid z).
$$

VAE 的最终目标是利用训练数据学习参数 $\phi$ 和 $\theta$，使模型分布 $p_\phi(x)$ 尽可能逼近真实数据分布。VAE 作为生成模型，要从隐空间生成有意义的结果，这就是通常所说的解码过程。将这个过程写成 $x\mid z$ 这样的条件概率形式：

$$
p_\phi(x)
= \int p_\phi(x\mid z)p(z)\,dz
= \mathbb{E}_{z\sim p(z)}\,p_\phi(x\mid z).
$$

训练时，我们想最大化训练集的对数似然，目标是：

$$
\max_\phi \log \prod_{j=1}^{N} p_\phi(x_j).
$$

但是这个积分通常很难直接计算，所以 VAE 不直接优化 $\log p_\phi(x)$，而是优化它的一个下界 ELBO。为什么是优化这个下界，下界的形式是什么，后面可以看到。下面需要对目标 $p_\phi(x)$ 进行一系列变形。

由贝叶斯公式：

$$
p_\phi(x,z)
= p_\phi(x\mid z)p(z)
= p_\phi(z\mid x)p_\phi(x).
$$

因此，

$$
p_\phi(x)
= \frac{p_\phi(x\mid z)p(z)}{p_\phi(z\mid x)}.
$$

目标转化为最大化 $\log p_\phi(x)$。将 $\log p_\phi(x)$ 写开，连等式的第一个等号的变换比较 tricky：

$$
\log p_\phi(x) \tag{1}
$$

$$
= \mathbb{E}_{z\sim q_\theta(z\mid x)}\log p_\phi(x) \tag{2}
$$

$$
= \mathbb{E}_{z\sim q_\theta(z\mid x)}
\log \frac{p_\phi(x\mid z)p(z)}{p_\phi(z\mid x)} \tag{3}
$$

$$
= \mathbb{E}_{z\sim q_\theta(z\mid x)}\log p_\phi(x\mid z)
+ \mathbb{E}_{z\sim q_\theta(z\mid x)}
\log \frac{p(z)q_\theta(z\mid x)}{p_\phi(z\mid x)q_\theta(z\mid x)} \tag{4}
$$

$$
= \mathbb{E}_{z\sim q_\theta(z\mid x)}\log p_\phi(x\mid z)
+ \mathbb{E}_{z\sim q_\theta(z\mid x)}
\log \frac{p(z)}{q_\theta(z\mid x)}
+ D_{\mathrm{KL}}\left(q_\theta(z\mid x)\,\Vert\,p_\phi(z\mid x)\right). \tag{5}
$$

从式（1）到式（2）这一步之所以成立，不是因为做了什么近似，而是因为 $\log p_\phi(x)$ 与 $z$ 无关。对任意关于 $z$ 的概率分布 $r(z\mid x)$，都有

$$
\log p_\phi(x)
= \mathbb{E}_{z\sim r(z\mid x)}\left[\log p_\phi(x)\right].
$$

其中式（5）的前两项就是 Evidence Lower Bound（ELBO）：

$$
\mathcal{L}_{\mathrm{ELBO}}
= \mathbb{E}_{z\sim q_\theta(z\mid x)}\log p_\phi(x\mid z)
+ \mathbb{E}_{z\sim q_\theta(z\mid x)}
\log \frac{p(z)}{q_\theta(z\mid x)} \tag{6}
$$

$$
= \underbrace{\mathbb{E}_{z\sim q_\theta(z\mid x)}\log p_\phi(x\mid z)}_{\text{Reconstruction Term}}
- \underbrace{D_{\mathrm{KL}}\left(q_\theta(z\mid x)\,\Vert\,p(z)\right)}_{\text{Latent Regularization}}. \tag{7}
$$

由于

$$
D_{\mathrm{KL}}\left(q_\theta(z\mid x)\,\Vert\,p_\phi(z\mid x)\right) \geq 0,
$$

所以

$$
\log p_\phi(x) \geq \mathcal{L}_{\mathrm{ELBO}}(x).
$$

上式是 Lower Bound 这个名字的来源。最大化 ELBO 可以起到最大化 $\log p_\phi(x)$ 的作用。实际上，最大化 ELBO 有两层含义：一是让 $\log p_\phi(x)$ 的下界本身变大，也就是让模型对数据的解释能力更强；二是让近似后验 $q_\theta(z\mid x)$ 更加接近真实后验 $p_\phi(z\mid x)$。

注意到式（5）的前两项，在 Gaussian VAE 中，编码器 $q_\theta(z\mid x)$ 通常建模为高斯分布，即

$$
q_\theta(z\mid x)
:= \mathcal{N}\left(z;\mu_\theta(x),\operatorname{diag}\left(\sigma_\theta^2(x)\right)\right),
$$

其中

$$
\mu_\theta:\mathbb{R}^D\to\mathbb{R}^d,
\qquad
\sigma_\theta:\mathbb{R}^D\to\mathbb{R}_+^d
$$

是编码器网络的确定性输出。解码器通常建模为具有固定方差的高斯分布，即

$$
p_\phi(x\mid z)
:= \mathcal{N}\left(x;\mu_\phi(z),\sigma^2I\right),
$$

其中

$$
\mu_\phi:\mathbb{R}^d\to\mathbb{R}^D
$$

是一个神经网络，而 $\sigma>0$ 是控制方差的一个较小常数。先验分布为

$$
p(z)=\mathcal{N}(0,I).
$$

现在注意式（5）的第三项：

$$
D_{\mathrm{KL}}\left(q_\theta(z\mid x)\,\Vert\,p_\phi(z\mid x)\right).
$$

将

$$
\mathbb{E}_{x\sim p_{\mathrm{data}}(x)}
\left[
D_{\mathrm{KL}}\left(q_\theta(z\mid x)\,\Vert\,p_\phi(z\mid x)\right)
\right] \tag{8}
$$

称为 Inference Error，衡量的是用编码器生成的后验分布 $q_\theta(z\mid x)$ 和真实后验分布 $p_\phi(z\mid x)$ 的差距。

从式（1）到式（5），可以自然地看出 VAE 的编码器是如何被自然引入的。总结起来就是：要训练这个生成式模型 VAE，我们能观测到的只有 data distribution $x\sim p_{\mathrm{data}}(x)$。将数据输入一个由参数 $\theta$ 参数化的神经网络，得到 $x$ 在隐空间中的表示 $z$。$z$ 通过一个由 $\phi$ 参数化的神经网络得到输出结果 $x'$。训练神经网络的参数 $\theta$ 和 $\phi$，使得 ELBO 最大，从而最大化目标分布：

$$
p_\phi(x)
= \int p_\phi(x\mid z)p(z)\,dz.
$$

这里要对所有 latent variable $z$ 积分。在深度生成模型里，这个积分通常没有解析解，或者计算代价极高。为了处理这个困难，我们人为引入一个分布

$$
q_\theta(z\mid x)
$$

去近似真实后验

$$
p_\phi(z\mid x).
$$

然后对 $\log p_\phi(x)$ 有了上面一系列变形。
