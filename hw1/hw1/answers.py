r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1.  The test-set is used in order to estimate our model performance in an **unbiased** and **neutral** way. Meaning it will be tested and evaluated 
    based on **unseen** and **unlearnt** samples. In fact, that is precisely our module goal - to label correctly as possible 
    unseen samples of data. Any use of samples from the training or validation data for evaluating our model performance will 
    dirty its score and will result in inaccurate measurement, since that data was initially used for training and tuning our model
    parameters.
2.  Assuming the given data set consists of shuffled, labeled samples distributed in a way resembles to the distribution
    of them in the real world, the entire data set can be divided into 2 independent parts :
    [x%] from the entire set for test and [(100-est set in tx)%] for test sets.
    The importance of choosing the that way is that we want to evaluate our model in the best way we can simulate
    the real world. When choosing to split the data we face a trade off: choosing too much data for the test set will give
    us better evaluation of our model performance (more close to reality), however it might cause our model to be trained poorer
    in the first place than it could've been, since it was trained on less data (more data was hidden from it during training).
    In many literatures on Machine Learning it is stated that the best split of data is around 30% for test and 70% for train.
3.  As explained in the previous sections, the test set is an independent set used **ONLY** in evaluating our model performance. Rest of the points
    in the ml pipeline is used for training and building our module, hence they must not have any intersection with the test set.
         
"""

part1_q2 = r"""
The answer is YES. We need to split some part of the training set as a validation set as part of the training process.
During the model training we need to tune and adjust our model parameters, in order to improve our model accuracy results.
In simple words the model needs be able to evaluate how well it is "learning", and how to "learn from its own mistakes".
It would be wrong to use the test set for that purposes, since the whole idea of the test set is to be independent and
unrelated to the learning process. Involving the test set in any of the learning steps, will result in bettering
with the test data prediction scores in the final test step, since we trained it that way from the first place by
allowing the model to have access to the test data. 
"""

# ==============
# Part 2 answers

part2_q1 = r"""
In theory, increasing k can lead in improving generalization for unseen data. In reality, the more nearest
neighbours from the same class an object has, the more accurate will be our classification. In our case, 
assigning a label to a digit picture number, based on $K$ other similiar digit pictures labeled 
with the same label is a more accurate classification than a classification of the same label based on $k {<<} K$
samples. 
However in many cases it is not a simple task to implement as it can seen from the upper graph.
Our main problem with increasing k is that our train set is finite, therefore each class has a finite number of samples.
Hence increasing k may lead to overlapping between classes and in addition of noise in some sense to the prediction, 
making it less accurate than it couldve been. For example if we choose $k$ to be as big as the size of the train set
the module will simply return the most common class from the data set for each sample we provide it as an input.
"""

part2_q2 = r"""
1.  The first problem with the described method is that it is not maximizing the full potential of the training set.
    If we define our training set as our validation set as well, then for each sample in the validation set, exist
    the same sample in the training set such that the distance from it is 0 (by definition of our metric system),
    which will always appear in the $k$-nearest neighbours set (for any $k$ we choose), providing absolutely no new
    information and interfering the prediction process.
    The 2nd problem with that method is that choosing our hyper parameters is directly effected by the validation set,
    meaning, defining the entire train set as validation set will result in over fitting to our train set.

2.  Same as explained in the first case - defining the test set as our validation data, will result in over fitting to
    the test set. By doing so, it will miss the main goal of the module, which is to classify as much as accurate as
    possible new and unseen data.
    In addition to that, as explained in previous parts, the test set must be **unseen** and **hidden** from the module 
    during its training, and it is meant to provide an **unbiased** way to measure our model performence in the final 
    step of the training process. Using the test set as validation set eliminates entirely the 'unseen', 'hidden', 
    and 'unbiased' parts, since the module was trained in a way to provide the best results for the test-set itself.
    
    Furthermore we can add that using the cross validation gives us the average validation error for each $k$,
    which provides us better information about the chosen $k$, rather than possible luck (or less luck) due to
    validating on the train or the test sets once for each $k$. Another thing is the addition of another dimension of
    randomization to our validating groups, since it generates new data every time and in some sense we can say it
    behaves as unseen and independent data in each of the iterations, which result in a more accurate results rather
    than generating one time, a fixed and constant validation set as described in both above cases.
    
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The selection of $\Delta > 0$ is arbitrary for the SVM loss $L(\mat{W})$ because in some sense is meaningnles,
since the weights can shrink or stretch the differences between the predicted and the g.t scores arbitrarily.
During our computation, the scores and their differences are directly effected by the magnitude of the weights, causing 
bigger values to increase the differences, while smaller values to decrease the differences eliminating the selection
of the delta in the first place.

In addition to the weights, lambda scalar may decrease or increase changes in  $\Delta$ so in the case we get to
choose both of them the selection of $\Delta > 0$ is arbitrary for the SVM loss.
"""

part3_q2 = r"""
1.  Given the images in the visualization section above we can interpret that the linear model is learning
    about the general differences between the brightness of the pixels for each of the digits. In simple words 
    the model tries to learn the way each of the digits are generally being written based on their structure and shapes
    (for example - the model understands that '0' is a kind of a circle, while '7' is an upper horizontal line, and a 
    diagonal line starting from the upper line right end, descending to the left side).
    Based on that we can better understand the some of the classification errors. For instance - 
    In the first line the model misclassified '5' with '6' since that version of '5' was written in a confusing way:
    The down part of the half circle, which is generally the way '5' is written, appears as a complete circle, more
    similiar to the way '6' is generally written.
    In the 6th line we can see in the 2nd picture, the model misclassified '6' with '2' since '6' was written in a 
    spiral way similiar to the way many '2's are being written.
    Furthermore we can see that while the model struggles with some digits (e.g '2','5','6') due to similiarity
    in some cases between them, it classifies other much "simpler" digits pretty good, like '0' (full circle) and '9'
    (upper circle and a down right line)  

2.  The similiarity between the KNN and the linear classifier is that they are both basing their decisions on
    an earlier seen and known labeled data, and they both use deductive reasoning in making conclusions about 
    each single input sample based on the model knowledge and its understanding about the world.
    The differences between them is the approach they both take in achieving their purposes. 
    The linear classifier approach is : "If it looks like a duck, based on my understanding of how ducks 
    should look like in general, then it is probably a duck". 
    While the KNN approach is: "If it looks similiar to other ducks that I saw earlier, then it is probably a duck". 
"""

part3_q3 = r"""
1.  Based on the graph of the training set (while trying to stay unbiased answering as much as I can:) )
    I would say that the learning rate I chose is pretty good, since it first complies with the assignment
    restrictions. Second we can see that overall trend of the learning accuracy is accending while the 
    trend of the loss is descending in a "natural" and "logical" way. The more epochs pass the more the model
    gets accurate.
    Choosing too high learning rate will cause in a too much diverging of the weights values, which will most likely
    cause in missing out our stationary point that it's trying to achieve.
    On the other hand choosing too low learning rate will cause in a too "slow" learning pace, resulting in poor results
    due to running out of time.

2.  Based on the graph of the training and test set accuracy we can see that from some point The model accuracy on the 
    train set is slightly better than the its accuracy on the validation set. Hence we can assume that the model is
    slightly over fitted to the training set.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern to see in a residual plot is a linear horizontal line crossing Y axis at 0. In that case
for every predicted value $\hat{y}^{(i)}$ the error is $y^{(i)} - \hat{y}^{(i)}=0$. Based on the residual plots above 
we can say that the error std of our module is ~4. Comparing to the top-5features module plot, we can see that we got 
better results since the top-5features module error std is ~5. 
"""

part4_q2 = r"""
1.  Once the data have been transformed into the new higher space it is able once again to search for a linear 
    separating hyperplane. Meaning, it is still a linear regression model, since we are still trying to solve
    a linear equation: $Z=WX+b$
2.  Yes. we can fit any non-linear function of the original features with this approach.
    We can always use "taylor's series" and we can always use feature engineering in order to extend our original dimension
    to a higher one.
3.  Adding of non-linear features to our data allows us to be able to better seperate data which is linearly 
    inseparable. By doing so, we transform our original input data into a higher dimensional space. 
    In such case extending our original D-dimensional data into D'-dimensional data will result in a hyperplane
    of (D'-1) dimensions.
"""

part4_q3 = r"""
1.  Np.logspace is a logarithmic scale which allows our to search a value in a wider range of values but
    with less number of samples, in contrast to the np.linearspace which is linear and the distance between 
    samples is constant.
    That way we can fit the parameters better to our model.
    
2.  without CV we try to fit each degree with each lambda so we make total fittings of $(num of \lambda samples) * (num of degree samples)$
    With CV we do the same but also we have to multiply by the number of num_folds from the cross validation.
    Therefore in that case we make total of $20*3*3$ = 180 fittings.
"""

# ==============
