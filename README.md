# BELS
Broad Ensemble Learning System (BELS)

In a data stream environment, classification models must efficiently and effectively handle
concept drift. Ensemble methods are widely used for this purpose; however, the ones available in the
literature either use a large data chunk to update the model or learn the data one by one. In the former,
the model may miss the changes in the data distribution, while in the latter, the model may suffer from
inefficiency and instability. To address these issues, we introduce a novel ensemble approach based on
the Broad Learning System (BLS), where mini chunks are used at each update. BLS is an effective
lightweight neural architecture recently developed for incremental learning. Although it is fast, it requires
huge data chunks for effective updates, and is unable to handle dynamic changes observed in data streams.
Our proposed approach named Broad Ensemble Learning System (BELS) uses a novel updating method
that significantly improves best-in-class model accuracy. It employs an ensemble of output layers to
address the limitations of BLS and handle drifts. Our model tracks the changes in the accuracy of the
ensemble components and react to these changes. We present the mathematical derivation of BELS, perform
comprehensive experiments with 35 datasets that demonstrate the adaptability of our model to various drift
types, and provide hyperparameter and ablation analysis of our proposed model. Our experiments show that
the proposed approach outperforms 10 state-of-the-art baselines and supplies an overall improvement of
18.59% in terms of average prequential accuracy.
