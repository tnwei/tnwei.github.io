---
title: "Explaining negative R-squared postscript"
date: 2022-06-06T00:36:00+08:00
draft: true
summary: ""
tags: ["statistics"]
math: true
---
# Post script

After writing the previous post XXX, I’m interested in figuring out the mathematical properties required for R2 between 0 and 1 to hold.

Some random statistical expert has the answer at [https://stats.stackexchange.com/a/183279/346912](https://stats.stackexchange.com/a/183279/346912).

Wrote up the math:

{{< figure src="/images/r2-bounded-condition.jpeg" alt="When is R2 not between 0 and 1" caption="When is R2 not between 0 and 1">}}

The explanation asserts that least squares multilinear regression with an intercept is guaranteed to have the extra expression to be zero, but didn’t go into detail. I expanded it above and arrived at the point where the sum of predictions need to equal the mean times number of data points for the expression to go to zero. I'm guessing that allowing an intercept grants the extra degree of freedom necessary in least squares regression for that to happen. Commented that under the question.

Not all questions are fully answered but I’m satisfied with what I learnt.
