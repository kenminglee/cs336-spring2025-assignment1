# Problem (`learning_rate`): Tune the learning rate

<img src="../images/Training Loss Curve for different LRs.png" alt="Alt text" style="width:80%;">
<img src="../images/Validation Loss Curve for different LRs.png" alt="Alt text" style="width:80%;">

The above hyperparameter search was performed based on a grid search over several values.
As can be observed by the above training loss and validation loss curve for various learning rates, we observe a few interesting observations:

- Generally speaking, lower training loss results in a lower validation loss. This indicates that our model isn't massively overfitting.
- Interestingly, not only does the LR affect speed of convergence, it also affects the final result achieved.
A LR of 5e-4 or 1e-3 resulted in the lowest final training loss and validation loss, followed by 0.01, 0.1 and 0.5, which converged to a suboptimal solution. 
At LR=1, the model fails to converge.
- Due to the small batch size (32), our gradient estimates are likely noisy, therefore a smaller learning rate leads to faster decrease in training loss, and also result in a lower final training/validation loss.