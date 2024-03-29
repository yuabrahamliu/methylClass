% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/multiomics.R
\name{maincv}
\alias{maincv}
\title{Cross validation on model performance}
\usage{
maincv(
  y.. = NULL,
  betas.. = NULL,
  cv.betas.path,
  cv.betas.prefix,
  subset.CpGs = 10000,
  n.cv.folds = 5,
  nfolds.. = NULL,
  K.start = 1,
  k.start = 0,
  K.stop = NULL,
  k.stop = NULL,
  seed = 1234,
  out.path,
  out.fname = "CVfold",
  cores = 10,
  topfeaturenumber = 50000,
  normalcv = FALSE,
  savefeaturenames = FALSE,
  method = "eSVM",
  anno = NULL,
  lasso = 3.25e-07,
  ridge = 0,
  n_pcs = 100,
  perplexity = 10,
  savefigures = FALSE,
  pythonpath = NULL,
  confoundings = NULL,
  padjcut = 0.05,
  xcutoff = 0.1,
  cutnum = 10000,
  ntrees = 500,
  p = 200,
  modelcv = 5,
  C.base = 10,
  C.min = -3,
  C.max = -2,
  learnernum = 10,
  minlearnersize = 1000,
  viewstandard = NULL,
  platform = "450K",
  max_depth = 6,
  eta = c(0.1, 0.3),
  gamma = c(0, 0.01),
  colsample_bytree = c(0.01, 0.02, 0.05, 0.2),
  subsample = 1,
  min.chwght = 1,
  nrounds = 100,
  early_stopping_rounds = 50,
  alpha.min. = 0,
  alpha.max. = 1,
  by. = 0.1,
  predefinedhidden = NULL,
  maxepochs = 10,
  activation = "Rectifier",
  momentum_start = 0,
  rho = 0.99,
  gridsearch = FALSE,
  num_epoch_pretrain = 500,
  num_epoch = 2500,
  adj_parameter = 10,
  dim_he_list = c(400, 400, 200),
  lr_e_pretrain = 0.001,
  lr_e = 5e-04,
  lr_c = 0.001,
  multiomicsnames = NULL,
  weighted = FALSE
)
}
\arguments{
\item{y..}{The true labels of the samples. Can be a vector, factor, or NULL.
If it is a vector or factor, each element is a label for a sample and the
element order in it should be the same as the sample order in the sample
data provided by the parameter \code{betas..}. If it is NULL, there should
be a vector or factor named \code{y} in the global environment and the function
will load it and use it as the sample true labels.}

\item{betas..}{The beta value matrix of the samples. Each row is one sample
and each column is one feature. The function will divide the samples there
internally into different cross validation training and testing sets to
train and evaluate the model. This parameter can also be set as NULL, and
in this case, the function will find and load the results of the function
\code{cvdata}, which are the training and testing sets generated by it in
advance. The absolute path of the folder with these divided data should be
provided by the parameter \code{cv.betas.path}, and the name prefix of the
data file should be provided by the parameter \code{cv.betas.prefix}, same
as the value transferred to the parameter \code{out.fname} of the function
\code{cvdata}.}

\item{cv.betas.path}{If the parameter \code{betas..} is set as NULL, this
parameter is necessary to provide the folder path of the divided training
and testing sets generated by the function \code{cvdata}, and it can be an
absolute directory string or a relative one.}

\item{cv.betas.prefix}{If the parameter \code{betas..} is set as NULL, this
parameter is necessary to provide the name prefix of the divided training
and testing sets generated by the function \code{cvdata}, corresponding to
the value of the parameter \code{out.fname} of the function \code{cvdata}.}

\item{subset.CpGs}{The feature selection method. It can be a numeric number
such as 10000, and in this case, the top 10000 most variable features of
the methylation data of each training set will be selected as the features
to construct the machine learning model. It can also be the string "limma"
and then \code{limma} will be used to select the differential features
between a sample class and all other samples. The differential ones should
fulfill the condition that the adjusted p-value < \code{padjcut} (default
is 0.05), and its betas value difference between the sample class and all
other samples should be > \code{xcutoff}, or < -\code{xcutoff} (default is
0.1). After the differential features for each class have been found by
\code{limma}, all of them will be mixed together and ordered according to
the adjusted p-value and the absolute value of beta value difference, and
finally the top \code{cutnum} features will be selected and used as the
features (default value is 10000). Besides, if there are any confounding
factors in the dataset need to be removed, they can be provided by the
parameter \code{confoundings} and \code{limma} will select the features
after these confoundings have been adjusted via linear regression. This
\code{confoundings} should be provided as a vector with the names of the
confoundings in the meta data (the meta data are provided by the parameter
\code{anno}) as elements. As to \code{subset.CpGs}, it can also be set as
the string "SCMER", and then \code{SCMER} will be used to select features
from each training set. In this case, other parameters should be set well,
including \code{lasso}, \code{ridge}, \code{n_pcs}, \code{perplexity},
\code{savefigures}, \code{pythonpath}, \code{topfeaturenumber}. Because
\code{SCMER} is a method to select features able to preserve the original
manifold structure of the data after its feature selection via elastic net
regularization, most of these parameters are used to config the manifold
(\code{n_pcs} and \code{perplexity}) as well as elastic net (\code{lasso},
\code{ridge}), while another important parameter is \code{pythonpath} that
is used to tell the function the absolute path of \code{Python} you want
to use to run \code{SCMER} because this method depends on \code{Python}.
Also, because \code{limma} and \code{SCMER} can be time-consuming for a
large number of candidate features, it is recommended to do a prescreen on
the data before running them, and if the parameter \code{betas..} is not
NULL, the parameter \code{topfeaturenumber} can be set as a numeric value
such as 50000, so that before internally dividing the samples into the
training and testing sets, the top 50000 most variable features will be
selected on the whole dataset first, and then, after obtaining the cross
validation training sets from it, the top variable, \code{limma}, or the
\code{SCMER} features can be selected on these prescreened sets. However,
if \code{betas..} is set as NULL and the training sets need to be loaded
from the results of \code{cvdata}, \code{topfeaturenumber} will not work
on them and the prescreened data need to be prepared during running the
function \code{cvdata}. The parameters \code{subset.CpGs} can also be set
as NULL, so that no feature selection will be done on the data before the
model construction.}

\item{n.cv.folds}{A numeric number and the default value is 5, so that if
the parameter \code{normalcv} is set as TRUE, a 5 fold cross validation
will be performed, and if \code{normalcv} is FALSE, a 5 by 5 nested cross
validation will be performed. This parameter only works if \code{betas..}
is not NULL and an internal training/testing sets division is needed.}

\item{nfolds..}{If the \code{n.cv.folds} parameter is provided as NULL, the
training/testing sets need to be divided following the cross validation
structure provided by this parameter and it is the result of the function
\code{makecv}, but if it is also NULL, and there is such a data structure
named "nfolds" in the environment, it will be loaded by the function and
use it to divide the training/testing sets. This parameter only works when
the \code{betas..} parameter is not NULL, because when it is NULL, this
cross validation structure can be fetched directly from the data loaded by
the parameters \code{cv.betas.path} and \code{cv.betas.prefix}. It is a
list with the sample indexes in the dataset showing which samples belong
to which training/testing sets.}

\item{K.start}{A number indicating from which outer cross validation loop
the model construction and evaluation should be started. Default is 1.}

\item{k.start}{A number indicating from which inner cross validation loop
the model construction and evaluation should be started. Default is 0, and
when the cross validation is a normal one, rather than a nested one, this
value should be 0.}

\item{K.stop}{The default of this value is NULL and indicates that the outer
cross validation loops will be used to training and evaluate the model
until all the loops have been finished. It can be also a number such as 3,
so that the process will stop when the outer loop number has been 3.}

\item{k.stop}{The default of this value is NULL and indicates that the inner
cross validation loops will be used to train and evaluate the model until
all the loops have been finished. It can also be a number such as 3, so
that this process will stop when the inner loop number has been 3, and for
a normal cross validation without inner loops, this parameter should be
NULL.}

\item{seed}{Some process performed by this function will need a seed number
to fix the random process, such as the training/testing data division, the
random sampling steps of some models such as random forest, eSVM, eNeural,
etc, and this parameter is used to set their seeds. Default value is 1234.}

\item{out.path}{For all the cross validation loops, their result files will
be saved in the folder set by this parameter. It is the folder name will
be created in the current working directory, so a relative not absolute
path.}

\item{out.fname}{The final cross validation result files will be saved in
the folder set by the parameter \code{out.path} and the name prefix of the
result files need to be set by this parameter, default is "CVfold".}

\item{cores}{The core number need to do parallelization computation. Default
is 10.}

\item{topfeaturenumber}{As mentioned in the \code{subset.CpGs} parameter
part, it is used to set the prescreened feature number when \code{betas..}
is not NULL. Default value is 50000. It can also be set as NULL, so that
no precreen will be done on the data.}

\item{normalcv}{Indicating whether the cross validation loops are normal or
nested. Default is FALSE, meaning nested cross validation.}

\item{savefeaturenames}{Default is FALSE, but if is set as TRUE, the feature
names selected by the feature selection process will be saved as a vector
for each cross validation loop.}

\item{method}{Which algorithm need to be used to train the model. Can be a
string as "RF", "SVM", "XGB", "ENet", "eNeural", "MOGONET", or "eSVM". The
default value is "eSVM".}

\item{anno}{A data frame recording the meta data of the samples, and should
contain at least 2 columns named as "label" and "sentrix". The former one
records the sample labels while the latter one records the sample IDs that
also used as row names of the methylation data matrices. The default value
is NULL and it is not necessary as long as the \code{y..} parameter is
provided, but if need to use \code{limma} to do the feature selection and
remove the confounding factors, it should be provided with the confounding
factors included in it.}

\item{lasso}{A parameter special for \code{SCMER} feature selection and it
defines the strength of L1 regularization in the elastic net process of
\code{SCMER}. Default is 3.25e10-7, so that around 10000 features will be
selected from 50000 prescreened candidate features.}

\item{ridge}{A parameter special for \code{SCMER} feature selection and it
defines the strength of L2 regularization in the elastic net process of
\code{SCMER}. Default is 0, so that the elastic net process is actually
a LASSO process.}

\item{n_pcs}{Number of principle components need to reconstruct the sample-
sample distance matrix during the \code{SCMER} selection. Default is 100.}

\item{perplexity}{Perplexity of tSNE modeling for the \code{SCMER} feature
selection. Default is 10.}

\item{savefigures}{Whether save the PCA and UMAP figures generated by the
\code{SCMER} method or not. Choose from TRUE and FALSE. Default is FALSE.}

\item{pythonpath}{Because the feature selection method \code{SCMER} and the
model training algorithm \code{MOGONET} are \code{Python} based methods,
the directory of the \code{Python} interpreter you want to use to run them
should be provided via this parameter, and to run \code{SCMER}, several
modules should be installed to the \code{Python} environment, including
\code{time}, \code{functiontools}, \code{abc}, \code{torch}, \code{numpy},
\code{typing}, \code{pandas}, \code{matplotlib}, \code{multiprocessing},
\code{scanpy}, and \code{sklearn}. To run \code{MOGONET}, the modules are
\code{numpy}, \code{sklearn}, and \code{torch}.}

\item{confoundings}{A parameter special for \code{limma} feature selection.
Details can be seen in the \code{subset.CpGs} parameter section.}

\item{padjcut}{A parameter for \code{limma} feature selection. Default value
is 0.05 and details can be seen in the \code{subset.CpGs} section.}

\item{xcutoff}{A parameter for \code{limma}. Details can also be seen in the
\code{subset.CpGs} section. Default value is 0.1.}

\item{cutnum}{A parameter special for \code{limma}. Details can be seen in
the \code{subset.CpGs} section.}

\item{ntrees}{A parameter special for the random forest (RF) model, defining
the number of decision trees in the RF model. Default is 500.}

\item{p}{A parameter special for RF. In the RF method here, a 2-step process
is conducted. The first one constructs an RF model on all the candidate
features with \code{ntrees} trees. Then, the top \code{p} most important
features in this step will be selected by calculating their influence on
the error of each tree using permuted out-of-bag data, and these features
will be transferred to the second step to construct a second RF model on
them, also with a tree number of \code{ntrees}. The parameter \code{p} is
used to control how many top important features are needed to be selected,
and the default value is 200.}

\item{modelcv}{For the models of SVM, eSVM, XGBoosting (XGB), elastic net
(ENet) and eNeural, a hyperparameter search step is performed to find the
optimal hyperparameters, via cross validation, such as the regularization
constant of SVM, eSVM, and ENet, and this parameter is used to define the
number of cross validation loops for hyperparameter search. Default is 5,
and it means to train a model from the training set of each model training
cross validation loop, this training set will be divided into 5 sets and
a 5-fold cross validation will be used to evaluate the performance of the
models with different hyperparameters and finally chooses the optimal one.
Hence, a model training cross validation loop, such as the normal cross
validation and nested cross validation loop, will further contain 5 cross
validation loops for hyperparameter search.}

\item{C.base}{A parameter special for SVM and eSVM to set the regularization
constant. This constant will be calculated by the function as base^index,
and \code{C.base} here serves as the base number. Combined with other 2
parameters \code{C.min} and \code{C.max} serving as indexes, it defines a
regularization constant series. Its start is \code{C.base}^\code{C.min},
and the end is \code{C.base}^\code{C.max}, while the near elements of the
series have a difference of \code{C.base} fold. If the 2 indexes are set
as the same, the series will become 1 regularization constant. The default
value of \code{C.base} is 10.}

\item{C.min}{As mentioned in the \code{C.base} part, this parameter is used
as the index of the small regularization constant number to set a series
for SVM and eSVM. Default is -3.}

\item{C.max}{As mentioned in the \code{C.base} part, this parameter is used
as the index of the large regularization constant number to set a series
for SVM and eSVM. Default is -2.}

\item{learnernum}{A parameter special for eSVM, eNeural and \code{MOGONET}
to set their base learner number. Default is 10.}

\item{minlearnersize}{A parameter special for eSVM, eNeural, \code{MOGONET}
to define the lower limit of the feature number of their base learners.
Default value is 1000, meaning each base learner should have at least 1000
features after the random sampling process to sample features for them.}

\item{viewstandard}{When this parameter is set as NULL. The features will be
assigned to the base learners of eSVM, eNeural and \code{MOGONET} through
random sampling. While if it is "Relation_to_Island" and the features are
DNA methylation probes, they will be split into groups of island probes,
N shelf and N shore probes, S shelf and S shore probes, and opensea probes
and then for each base learner, its features will be sampled from one of
these groups. If this parameter is set as "UCSC_RefGene_Group", then the
probes will be grouped into promoter probes, gene body probes and other
probes and each base learner will get its features via sampling on one of
these groups. The default value of this parameter is NULL.}

\item{platform}{When \code{viewstandard} is set as "Relation_to_Island" or
"UCSC_RefGene_Group", this parameter will be used to define the platform
of the probe annotation information to split them into different groups.
The default value is "450K", and can also "EPIC".}

\item{max_depth}{A parameter special for XGB. Its the maximum depth of each
tree. Default is 6.}

\item{eta}{A parameter special for XGB. It controls the learning rate via
scaling the contribution of each tree by a factor of 0 < eta < 1 when the
tree is added to the approximation, and can prevent overfitting by making
the boosting process more conservative. Its default value is a vector of
\code{c(0.1, 0.3)}, meaning a grid search will be conducted between these
2 values to find the optimal \code{eta} value with less misclassification
rate.}

\item{gamma}{A parameter special for XGB. Defines the minimum loss reduction
required to make a further partition on a leaf node of the tree. Default
value is a vector of \code{c(0, 0.01)} and a grid search will be conducted
on it.}

\item{colsample_bytree}{Special for XGB. Defines subsample ratio of columns
when constructing each tree. Default is \code{c(0.01, 0.02, 0.05, 0.2)},
and a grid search will be performed on it.}

\item{subsample}{Subsample ratio of the training instance for XGB. Setting
it to 0.5 means that XGB randomly collected half of the data instances to
grow trees and this can prevent overfitting and make computation shorter.
Its default value is 1.}

\item{min.chwght}{Minimum sum of instance weight (hessian) needed in a child
of the tree in XGB. If the tree partition step results in a leaf node with
the sum of instance weight less than this value, the building process will
give up further partitioning. Default is 1.}

\item{nrounds}{A parameter special for XGB. Defines the max number of the
boosting iterations. Default is 100.}

\item{early_stopping_rounds}{Special for XGB and default value is 50, which
means the training with a validation set will stop if the performance dose
not improve for 50 rounds.}

\item{alpha.min.}{A parameter special for ENet. Need to use with the other 2
parameters \code{alpha.max.} and \code{by.} to set an elastic net mixing
parameter series. The default value of \code{alpha.min.} is 0, the default
value of \code{alpha.max.} is 1, and the default value of \code{by.} is
0.1, so that a mixing parameter series staring with 0, ending with 1, and
with a difference between its neighbor elements as 0.1 will be generated
and to do a grid search on it to select the optimal mixing parameter value
(alpha) giving the smallest MSE across the hyperparameter searching cross
validation and then it will be used for the next model training.}

\item{alpha.max.}{A parameter special for ENet. Need to use with the other 2
parameters \code{alpha.min.} and \code{by.} to set an elastic net mixing
parameter series. As mentioned in the \code{alpha.min.} section.}

\item{by.}{A parameter special for ENet. Detail is in the \code{alpha.min.}
section.}

\item{predefinedhidden}{A parameter special for eNeural. Use it to transfer
the node number of each hidden layer of one neural network in the eNeural
model. Need to be a vector, such as \code{c(100, 50)}, so that for each
neural network in the eNeural model, 2 hidden layers will be set up. One
is with 100 nodes, while the other is with 50 ones. Default value is NULL,
so that the function will set up a hidden layer structure automatically.
If the parameter \code{gridsearch} is set as FALSE, this structure is with
2 layers and the node number of them are both around 1/100 of the input
node number. If \code{gridsearch} is TRUE, several different hidden layer
structures will be generated and a search will be performed on them to get
the optimal one.}

\item{activation}{Activation function special for eNeural. Can be a string
or a vector of strings to do grid search. Default is "Rectifier". Can also
be "Tanh" or "Maxout", or a vector with elements from them.}

\item{momentum_start}{Special for eNeural. Defines the initial momentum at
the beginning of training (try 0.5). Default is 0. And a vector covering
different values can be used for hyperparameter grid search.}

\item{rho}{Speical parameter for eNeural. Adaptive learning rate time decay
factor (defines the similarity to prior updates). Default is 0.99. Can be
a vector for hyperparameter grid search.}

\item{gridsearch}{Special parameter for eNeural. Whether do grid search to
select the optimal hyperparameters, or directly use the fixed and given
hyperparameters to train the neural networks. If it is TRUE, grid search
will be performed on the hyperparameters of hidden layer size and depth,
epoch number, activation function, initial momentum, and adaptive learning
rate time decay factor.}

\item{num_epoch_pretrain}{Special parameter for \code{MOGONET} and defines
the epoch number for its pretraining process. Default is 500.}

\item{num_epoch}{Special parameter for \code{MOGONET} and defines the epoch
number for its training process. Default is 2500.}

\item{adj_parameter}{Special parameter for \code{MOGONET} and defines the
the average number of edges per node that are retained in the adjacency
matrix used for graph convolutional networks (GCNs) construction. Default
is 10.}

\item{dim_he_list}{Special parameter for \code{MOGONET} and is to define the
node number of each hidden layer of the GCN network. Need to be a vector
with numbers as elements, such as \code{c(400, 400, 200)}, so that in each
GCN networks in \code{MOGONET}, 3 hidden layers will be set up. One is with
400 nodes, while the others are with 400 and 200 nodes. The default value
is \code{c(400, 400, 200)}.}

\item{lr_e_pretrain}{Special parameter for \code{MOGONET} and used to define
the learning rate of the GCN networks for the single-omics data at their
pretraining stage. Default value is 1e-3.}

\item{lr_e}{Special parameter for \code{MOGONET} and defines the learning
rate of the GCN networks for the single-omics data at the training stage.
Default value is 5e-4.}

\item{lr_c}{Special parameter for \code{MOGONET} and defines the learning
rate of the view correlation discovery network (VCDN) to aggregate the GCN
network results. Default is 1e-3.}

\item{multiomicsnames}{Used for multi-omics model training. In this case, a
matrix should be organized using rows as samples and columns as features,
and the features should come from all the omics data want to use. Then,
same as the methylation data, the function can receive the matrix via the
parameter \code{betas..} or via loading the training/testing sets division
results of the function \code{cvdata}. To demonstrate which features in
the matrix are from which omics, the parameter \code{multiomicsnames} need
to be used to transfer a vector to the function. The element order in the
vector should be the same as the feature order in the matrix. An element
is the omics names of one feature. The default value is NULL and in this
case the data will be treated as single-omics data, but if an omics name
indication vector is provided, the data will be treated as multi-omics.}

\item{maxepchs}{A parameter special for eNeural and defines the epoch number
for the neural network training. If the parameter \code{gridsearch} is set
as FALSE, the epoch number is fixed as this, but if \code{gridsearch} is
TRUE, an epoch number series will be set up starting from 10 and ending at
\code{maxepchs}, with the neighbor elements having a 10 fold difference.
Then, grid search will be performed across this series. The default value
of \code{maxepchs} is 10.}
}
\value{
The cross validation results will be saved in the directory set by
\code{out.path}, and each cross validation loop will have an .RData file
saved there, with the testing sample prediction score matrix, the trained
model object, the training and testing sample indexes of that loop, etc in
it. These results are the results from the raw model and are needed by the
calibration step conducted by the function \code{maincalibration}.
}
\description{
Cross validation on model performance without calibration.
}
\examples{
library(methylClass)

labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
labels <- readRDS(labels)

betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
betas <- readRDS(betas)

maincv(y.. = labels, betas.. = betas, subset.CpGs = 10000, n.cv.folds = 5, 
 normalcv = TRUE, out.path = 'RFCV', out.fname = 'CVfold', 
 method = 'RF', seed = 1234, cores = 4)
 
\dontrun{
maincv(y.. = labels, betas.. = betas, subset.CpGs = 10000, n.cv.folds = 5, 
 normalcv = TRUE, out.path = 'eSVMCV', out.fname = 'CVfold', 
 method = 'eSVM', seed = 1234, cores = 4)
}
}
