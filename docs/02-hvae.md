# Hierarchical VAE

**Yola25**  
**March 2026**

---

## 1. 回顾：VAE 的生成过程

回顾 VAE 作为生成模型，其本质生成过程是：

$$
z \sim p(z), \qquad x \sim p_\phi(x \mid z).
$$

从“生成”的角度，或者说从定义生成模型的角度，编码器并不是必须的。只需要：

- 一个先验分布 $p(z)$；
- 一个解码器 $p_\phi(x \mid z)$。

但是这样无法直接训练模型。因为训练时我们拿到的是数据 $x$，想要最大化：

$$
\log p_\phi(x)
=
\log \int p_\phi(x \mid z)p(z)\,dz.
$$

这里的困难在于：对于一个给定的真实样本 $x$，我们并不知道“什么样的 $z$”最有可能生成它。也就是说，我们需要知道：

$$
p_\phi(z \mid x),
$$

即“给定 $x$，它对应的 latent code 分布是什么”。但这个后验一般不可解。因此，自然地引入编码器。编码器是在看到样本 $x$ 后，猜测它在 latent space 里面最可能来自哪些 $z$，学习分布：

$$
q_\theta(z \mid x) \approx p_\phi(z \mid x).
$$

---

## 2. VAE 与 HVAE 的结构对比

单层 VAE 可以概括为：

```text
x  --encoder q_θ(z|x)-->  z  --decoder p_φ(x|z)-->  x
```

Hierarchical VAE 将单个隐变量 $z$ 扩展为多层隐变量 $z_1,z_2,\ldots,z_L$。其推断方向和生成方向可以概括为：

```text
x  --q_θ(z₁|x)-->  z₁  --q_θ(z₂|z₁)-->  z₂  --> ... -->  z_L
x  <--p_φ(x|z₁)--  z₁  <--p_φ(z₁|z₂)--  z₂  <-- ... <--  z_L
```

其中，编码器沿着从 $x$ 到高层 latent variable 的方向进行推断；生成模型则从最高层 latent variable $z_L$ 开始，逐层向下生成，最终生成观测变量 $x$。

---

## 3. 单层 VAE 的 ELBO

回顾在 Gaussian VAE 中已经给出的 $\mathcal{L}_{\mathrm{ELBO}}$：

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=
\mathbb{E}_{z \sim q_\theta(z \mid x)}
\big[\log p_\phi(x \mid z)\big]
-
D_{\mathrm{KL}}\big(q_\theta(z \mid x)\,\|\,p(z)\big).
$$

其中：

$$
\mathbb{E}_{z \sim q_\theta(z \mid x)}
\big[\log p_\phi(x \mid z)\big]
$$

是 **reconstruction term**，而

$$
D_{\mathrm{KL}}\big(q_\theta(z \mid x)\,\|\,p(z)\big)
$$

是 **latent regularization term**。

VAE 的训练目标是最大化 ELBO 在真实数据分布上的期望，即：

$$
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[\mathcal{L}_{\mathrm{ELBO}}(x)\big]
=
\mathbb{E}_{p_{\mathrm{data}}(x)q_\theta(z \mid x)}
\big[\log p_\phi(x \mid z)\big]
-
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,p(z))\big].
$$

其中第一项是重构项，刻画给定隐变量 $z$ 时解码器对样本 $x$ 的解释能力；第二项是正则项，约束编码器输出的后验分布不要偏离先验 $p(z)$ 过远。

---

## 4. Aggregated posterior 与 mutual information

为了进一步理解 KL 正则项的含义，引入 **aggregated posterior**：

$$
q(z)
=
\int p_{\mathrm{data}}(x)q_\theta(z \mid x)\,dx,
$$

以及由联合分布

$$
q(x,z)
=
p_{\mathrm{data}}(x)q_\theta(z \mid x)
$$

所定义的互信息：

$$
I_q(x;z)
=
\mathbb{E}_{q(x,z)}
\left[
\log \frac{q_\theta(z \mid x)}{q(z)}
\right]
=
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,q(z))\big].
$$

在这些定义下，ELBO 中的 KL 正则项可进一步分解为：

$$
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,p(z))\big]
=
I_q(x;z)
+
D_{\mathrm{KL}}(q(z)\,\|\,p(z)).
$$

因此，ELBO 的期望可以重写为：

$$
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[\mathcal{L}_{\mathrm{ELBO}}(x)\big]
=
\mathbb{E}_{p_{\mathrm{data}}(x)q_\theta(z \mid x)}
\big[\log p_\phi(x \mid z)\big]
-
I_q(x;z)
-
D_{\mathrm{KL}}(q(z)\,\|\,p(z)).
$$

这个式子表明，最大化 ELBO 不仅要求提高重构质量，还隐式地受到两个惩罚项的约束：

1. 样本 $x$ 与隐变量 $z$ 之间的互信息 $I_q(x;z)$；
2. aggregated posterior $q(z)$ 与先验 $p(z)$ 之间的偏离程度 $D_{\mathrm{KL}}(q(z)\,\|\,p(z))$。

这一分解也解释了当解码器过强时，模型可能倾向于压低 $I_q(x;z)$，从而导致 **posterior collapse**。

---

## 5. KL 正则项分解的推导

首先，根据 KL 散度定义：

$$
D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,p(z))
=
\mathbb{E}_{z \sim q_\theta(z \mid x)}
\left[
\log \frac{q_\theta(z \mid x)}{p(z)}
\right].
$$

再对 $x \sim p_{\mathrm{data}}(x)$ 取期望，有：

$$
\begin{aligned}
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,p(z))\big]
&=
\mathbb{E}_{p_{\mathrm{data}}(x)}
\left[
\mathbb{E}_{z \sim q_\theta(z \mid x)}
\left[
\log \frac{q_\theta(z \mid x)}{p(z)}
\right]
\right] \\
&=
\mathbb{E}_{q(x,z)}
\left[
\log \frac{q_\theta(z \mid x)}{p(z)}
\right],
\end{aligned}
$$

其中：

$$
q(x,z)=p_{\mathrm{data}}(x)q_\theta(z \mid x).
$$

接下来，在对数中乘除同一个 $q(z)$，得到：

$$
\log \frac{q_\theta(z \mid x)}{p(z)}
=
\log \frac{q_\theta(z \mid x)}{q(z)}
+
\log \frac{q(z)}{p(z)}.
$$

将上式代入，可得：

$$
\begin{aligned}
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,p(z))\big]
&=
\mathbb{E}_{q(x,z)}
\left[
\log \frac{q_\theta(z \mid x)}{q(z)}
\right]
+
\mathbb{E}_{q(x,z)}
\left[
\log \frac{q(z)}{p(z)}
\right].
\end{aligned}
$$

其中第一项按照定义正是互信息：

$$
\mathbb{E}_{q(x,z)}
\left[
\log \frac{q_\theta(z \mid x)}{q(z)}
\right]
=
I_q(x;z).
$$

对于第二项，由于被积函数仅依赖于 $z$，可将 $x$ 积掉：

$$
\mathbb{E}_{q(x,z)}
\left[
\log \frac{q(z)}{p(z)}
\right]
=
\mathbb{E}_{q(z)}
\left[
\log \frac{q(z)}{p(z)}
\right]
=
D_{\mathrm{KL}}(q(z)\,\|\,p(z)).
$$

因此：

$$
\mathbb{E}_{p_{\mathrm{data}}(x)}
\big[D_{\mathrm{KL}}(q_\theta(z \mid x)\,\|\,p(z))\big]
=
I_q(x;z)
+
D_{\mathrm{KL}}(q(z)\,\|\,p(z)).
$$

---

## 6. VAE 的缺陷：posterior collapse

如果 decoder 足够强，即使不用到隐空间变量 $z$，也能把数据分布 $p_{\mathrm{data}}(x)$ 拟合得很好：

$$
p_\phi(x \mid z) \approx p_{\mathrm{data}}(x).
$$

此时，ELBO 的最优解可能会将编码器设置为：

$$
q_\theta(z \mid x)=p(z).
$$

这意味着给定任何 $x$，编码器输出的都是同一个分布 $p(z)$。于是 $z$ 和 $x$ 独立，因而：

$$
I_q(x;z)=0.
$$

这就是 **posterior collapse**：后验塌缩到先验，latent code 不再携带关于数据 $x$ 的信息，于是这个 latent code 就不能再作为有意义的特征表示。

例如，原本希望相似图片的 latent code 在隐空间中是接近的，但是发生 posterior collapse 之后，latent code $z$ 只是从先验里随机采样出来的噪声，无法做到生成有意义的结果。

posterior collapse 不是因为网络太浅、容量不够，而是因为目标函数本身允许这种坏解存在。如果 decoder 足够强，那么忽略隐空间的信息本身是一个对 ELBO 很有利的策略。把网络加深，反而可能让 decoder 更强，更容易走向 collapse。

所以问题不是“VAE 模型不够强”，而是：模型强到不需要 latent variable 也能把数据拟合好。这是 VAE 的问题，无法简单通过更深层的网络来解决这个缺陷，因此需要更复杂的 hierarchical 架构。

---

## 7. 自然引入 Hierarchical VAE

与单层 VAE 类似，层次变分自编码器，即 **hierarchical variational autoencoder, HVAE**，同样通过对边缘对数似然

$$
\log p_\phi(x)
=
\log \int p_\phi(x,z_{1:L})\,dz_{1:L}
$$

构造一个可计算的下界来进行训练。

HVAE 的关键区别在于，它引入了多个层次的隐变量：

$$
z_{1:L}=(z_1,z_2,\ldots,z_L),
$$

从而将潜在表示组织成一个层次生成结构。

---

## 8. HVAE 的生成模型与推断模型

在层次模型中，联合分布通常写为：

$$
p_\phi(x,z_{1:L})
=
p_\phi(x \mid z_1)
\prod_{i=2}^{L}p_\phi(z_{i-1}\mid z_i)
p(z_L).
$$

这表明生成过程从最高层潜变量 $z_L$ 开始，逐层向下生成中间隐变量，最终由最底层隐变量 $z_1$ 生成观测变量 $x$。

与之对应，推断模型，即编码器，可写为：

$$
q_\theta(z_{1:L}\mid x)
=
q_\theta(z_1\mid x)
\prod_{i=2}^{L}q_\theta(z_i\mid z_{i-1}).
$$

---

## 9. HVAE 的 ELBO 推导

对边缘对数似然引入近似后验 $q_\theta(z_{1:L}\mid x)$，有：

$$
\begin{aligned}
\log p_\phi(x)
&=
\log \int p_\phi(x,z_{1:L})\,dz_{1:L} \\
&=
\log \int
\frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}
q_\theta(z_{1:L}\mid x)\,dz_{1:L} \\
&=
\log \mathbb{E}_{q_\theta(z_{1:L}\mid x)}
\left[
\frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}
\right] \\
&\geq
\mathbb{E}_{q_\theta(z_{1:L}\mid x)}
\left[
\log \frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}
\right] \\
&=:
\mathcal{L}_{\mathrm{ELBO}}(x),
\end{aligned}
$$

其中最后一步使用了 Jensen 不等式。

将生成模型与推断模型的分解形式代入，可得：

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=
\mathbb{E}_{q_\theta(z_{1:L}\mid x)}
\left[
\log
\frac{
p(z_L)
\prod_{i=2}^{L}p_\phi(z_{i-1}\mid z_i)
p_\phi(x\mid z_1)
}{
q_\theta(z_1\mid x)
\prod_{i=2}^{L}q_\theta(z_i\mid z_{i-1})
}
\right].
$$

进一步整理后，HVAE 的 ELBO 可以写成“重构项减去多层 KL 项”的形式：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x)
=&\;
\mathbb{E}_{q}
\big[\log p_\phi(x\mid z_1)\big] \\
&-
\mathbb{E}_{q}
\big[D_{\mathrm{KL}}(q_\theta(z_1\mid x)\,\|\,p_\phi(z_1\mid z_2))\big] \\
&-
\sum_{i=2}^{L-1}
\mathbb{E}_{q}
\big[D_{\mathrm{KL}}(q_\theta(z_i\mid z_{i-1})\,\|\,p_\phi(z_i\mid z_{i+1}))\big] \\
&-
\mathbb{E}_{q}
\big[D_{\mathrm{KL}}(q_\theta(z_L\mid z_{L-1})\,\|\,p(z_L))\big],
\end{aligned}
$$

其中：

$$
\mathbb{E}_{q}
:=
\mathbb{E}_{p_{\mathrm{data}}(x)q_\theta(z_{1:L}\mid x)}.
$$

---

## 10. HVAE 的核心特征

与单层 VAE 相比，HVAE 的一个核心特征是：**每一层推断分布都与其对应的生成分布直接对齐**。

具体来说：

- 最底层的推断分布 $q_\theta(z_1\mid x)$ 与生成分布 $p_\phi(z_1\mid z_2)$ 对齐；
- 中间各层的推断分布 $q_\theta(z_i\mid z_{i-1})$ 与生成分布 $p_\phi(z_i\mid z_{i+1})$ 对齐；
- 最顶层的推断分布 $q_\theta(z_L\mid z_{L-1})$ 与先验 $p(z_L)$ 对齐。

因此，单层 VAE 中由一个整体 KL 项承担的信息约束，在 HVAE 中被分散到了各个相邻层之间的 KL 项中。换言之，HVAE 将信息惩罚分布到不同层级。

这种层次化潜变量结构的意义在于：模型不再依赖单一隐变量同时承担所有语义信息，而是允许不同层次的潜变量学习不同尺度、不同抽象程度的表示。

- 较高层的潜变量更适合建模全局、抽象的语义特征；
- 较低层的潜变量更适合刻画局部、细节性的变化。

这种性质来源于潜变量图结构本身的层次化设计，而不是简单地把平坦 VAE 的神经网络加深。

也就是说，HVAE 的优势并不只是“网络更深”，而是“隐变量之间的概率依赖关系被重新组织成层次结构”，从而改变了 ELBO 中信息约束的分布方式。

---

## 11. 小结

单层 VAE 的核心结构是：

$$
z \sim p(z), \qquad x \sim p_\phi(x\mid z),
$$

并通过近似后验 $q_\theta(z\mid x)$ 构造 ELBO 进行训练。

但是当 decoder 过强时，模型可能出现 posterior collapse，即：

$$
q_\theta(z\mid x)=p(z), \qquad I_q(x;z)=0.
$$

此时 latent variable 不再携带关于样本 $x$ 的有效信息。

HVAE 通过引入多层潜变量：

$$
z_{1:L}=(z_1,z_2,\ldots,z_L),
$$

将原本单层 VAE 中的一个 KL 正则项分解为多个层级之间的 KL 对齐项，从而使不同层次的 latent variable 可以承担不同尺度的信息表示。
