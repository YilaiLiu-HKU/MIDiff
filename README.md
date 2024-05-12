# NetDiffus
This is repository of the paper [NetDiffus: Network Traffic Generation by Diffusion Models through Time-Series Imaging](https://arxiv.org/abs/2310.04429) .

# Requirements

- Python 3.9
- guided-diffusion
- torch
- tqdm
- blobfile>=1.0.5

# About NetDiffus

While Machine-Learning based network data analytics are now common-
place for many networking solutions, nonetheless, limited access to appropriate
networking data has been an enduring challenge for many networking problems.
Causes for lack of such data include complexity of data gathering, commercial
sensitivity, as well as privacy and regulatory constraints. To overcome these
challenges, we present a Diffusion-Model (DM) based end-to-end framework,
NetDiffus, for synthetic network traffic generation which is one of the emerg-
ing topics in networking and computing system. NetDiffus first converts one-
dimensional time-series network traffic into two-dimensional images, and then
synthesizes representative images for the original data. We demonstrate that
NetDiffus outperforms the state-of-the-art traffic generation methods based on
Generative Adversarial Networks (GANs) by providing 66.4% increase in the
fidelity of the generated data and an 18.1% increase in downstream machine
learning tasks. We evaluate NetDiffus on seven diverse traffic traces and show
that utilizing synthetic data significantly improves several downstream ML tasks
including traffic fingerprinting, anomaly detection and traffic classification.

![img.png](img.png)


# Acknowledgements
This code is developed on the OpenAI's [Guided Diffusion](https://github.com/openai/guided-diffusion).
