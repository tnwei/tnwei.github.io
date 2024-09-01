---
title: "IOAI 2024 MysteryNet Explained"
date: 2024-09-01T17:04:00+08:00
summary: "Hid something in these safetensors!"
tags: []
draft: false
math: false
---

{{< rawhtml >}}
<div style="text-align: center">
<video width="600" autoplay muted loop style="display: block; margin: auto">
  <source src="/images/msian-flag-progression.mp4" type="video/mp4">
</video>
  <p><small><em>
Selamat Hari Merdeka! :)
  </em></small></p>
</div>
{{< /rawhtml >}}


At the end of IOAI 2024, I sent a gift to all students that required some elbow grease to open.

{{<figure src="/images/msian-gift-announcement-screenshot.jpeg" alt="Screenshot of the gift announcement in Discord" >}}

All the students are given is a `safetensors` file by their country / team name, and also this network definition below for a particular `MysteryNet`:

```python
class MysteryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(in_features=2, out_features=16)
        self.dense2 = nn.Linear(in_features=16, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dense3 = nn.Linear(in_features=32, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(in_features=32, out_features=32)
        self.dense6 = nn.Linear(in_features=32, out_features=32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dense7 = nn.Linear(in_features=32, out_features=32)
        self.dense8 = nn.Linear(in_features=32, out_features=16)
        self.bn8 = nn.BatchNorm1d(16)
        self.dense_head = nn.Linear(in_features=16, out_features=3)

    def forward(self, x):
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.dense2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = self.dense3(x)
        x = F.gelu(x)
        x = self.dense4(x)
        x = F.gelu(x)
        x = self.bn4(x)

        x = self.dense5(x)
        x = F.gelu(x)
        x = self.dense6(x)
        x = F.gelu(x)
        x = self.bn6(x)

        x = self.dense7(x)
        x = F.gelu(x)
        x = self.dense8(x)
        x = F.gelu(x)
        x = self.bn8(x)

        x = self.dense_head(x)

        return x
```

Here's [the link](https://drive.google.com/drive/u/1/folders/1HbonYHOuCLRBPWQcC-ZaXQVBT_ry_nSX) to the weights and `mysterynet.py` if you want to try it out yourself.

# Explanation

`MysteryNet` has 2d inputs and 3d outputs. The dimension of the output itself is a hint that the output is in RGB space, but without any spatial priors owing to how the network architecture itself does not inherit any from the 2d inputs. With some trial and error, you might stumble upon using xy coordinates as the network input.

What I did is to over-fit the networks to the point of being a hashmap that takes in image coordinates and outputs RGB values. This concept is called a compositional pattern-producing network (CPPN), introduced by Ken Stanley in 2007. I have misappropriated the idea to perform a belabored, lossy compression of your respective flags ;)

This exercise was inspired by [this blogpost](https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/) by David Ha, dating back to 2016.

Here is a jumbo collage of what you should see if you formatted your input correctly:

{{< figure src="/images/ioai2024-msian-gift-collage.jpeg" alt="Collage of various flags hidden in CPPN weights!" caption="Pardon the rough edges, I was in a rush" align=center >}}


# Miscellanea

- The looping video at the top of the page is the training progression of the CPPN encoding the Malaysian flag.
- Apologies in advance if I wasn't able to replicate your nation's flag faithfully!
- All flags are found online and resized to have 320px width. Through this exercise I learnt that (1) there isn't a standardized width-to-height ratio for flags, and (2) some nations have flags that are taller than they are wide!
- Easiest flags: Japan, Bangladesh, Poland
- Hardest flags: Iran, USA
- Most challenging bit was figuring out the training recipe in time before the conclusion of the event. The CPPNs were having difficulty reproducing details in some of the more elaborate flags. Probably not helped by my insistence on using the smallest feasible architecture. 
- I eventually learnt to simply leave the networks to train. Turns out my chosen network architecture did have sufficient capacity to encode the flags, it was instead I who did not have enough patience. I was killing off training runs too early when I saw the losses plateau, but then I realized that it takes an exponential amount of time for the networks to get really small details right. Reminds me of this nugget of wisdom [by Karpathy](http://karpathy.github.io/2019/04/25/recipe/): 

> _leave it training. I’ve often seen people tempted to stop the model training when the validation loss seems to be leveling off. In my experience networks keep training for unintuitively long time. One time I accidentally left a model training during the winter break and when I got back in January it was SOTA (“state of the art”)_.

- The final training recipe is a three stage affair: (1) train the network on a Gaussian blurred flag with some random noise added, (2) train the network to reproduce the original flag with random noise added (but less), then finally (3) leave the network to finetune on the flag with no noise added. The learning rate is decreased linearly across the duration of training. The idea is to get the general shapes and colours right, before allowing the network to work on the fine-grained details of the image. In my mind, blurring helped avoid sharp boundaries develop in easier parts of the image way ahead of the rest of the image, while random noise helped the network to avoid local minima that are easier to get in but hard to exit when finetuning. Can't verify these hunches though, didn't have time to experiment. 
- Thoughts on why the fine details are challenging and hypotheses on how to make training faster: the dense nature of the network means that changing the output for a single coordinate affects the outputs of its neighboring coordinates. My (unvalidated) conjecture is introducing some form of sparsity such that the network behaves like an ensemble of subnets that do not interact with each other might help.
- Why safetensors: don't unpack pickle files from sources that you don't trust! Pytorch save uses pickle under the hood, which is vulnerable to arbitrary code execution.
- In hindsight, including some positional encoding might help the CPPNs train a lot better. But it would also make the network less mysterious, which would be less exciting.

# Acknowledgements

- The idea to do this arose from speaking to Chris (Romania) over making interesting olympiad questions.
- Thanks to Fredrik's (Sweden) interest in flags, I had someone to share my progress with as I was making this, which is grealy appreciated. 
- Thanks to my students for being my beta-testers.
