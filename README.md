
# Stellar Blend Image Classification

## Project Summary

Stellar blends are a challenge in visualizing celestial bodies and are typically disambiguated through expensive methods. To address this, we propose an automated pipeline to distinguish single stars and blended stars in low resolution images. We apply different normalizations to the data, which are passed as inputs into machine learning methods and to a computationally efficient Gaussian process model (MuyGPs). MuyGPs with $N^{th}$ root local min-max normalization achieves 86\% accuracy (i.e. 12\% above the second-best). Moreover, MuyGPs outperforms the benchmarked models significantly on limited training data. Further, MuyGPs low-confidence predictions can be redirected to a specialist for human-assisted labeling 

## Project Objectives
* Data exploration and visualization of the stellar blends
* Explore possible embedding/normalization before model fitting
* Binary image classification problem (blended vs. non-blended)
* Apply MuyGPs, and at least 2 other machine learning methods for image classification of the stellar blending identification


## Key questions to answer

* What do the differences in blended stars look like in real data?
* How does the performance of image classification with MuyGPs compare to other machine learning methods? (time, accuracy, etc)
* Does data normalization and/or embedding improve performance?
* Discussion: can we trust ML classification of stellar blending? How can we improve this in the future?

### Links

* [MuyGPs: Scalable Gaussian Process Hyperparameter Estimation Using Local Cross-Validation](https://arxiv.org/abs/2104.14581)

