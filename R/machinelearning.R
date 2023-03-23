
#Parallelization####

#Some parallel computing going on in the background that is not getting cleaned up
#fully between runs can cause Error in summary.connection(connection) : invalid
#connection. The following function is needed to be called to fix this error.
unregister_dopar <- function(){

  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)

}



mclapply.win <- function(X, FUN, ..., mc.cores){

  cl <- parallel::makeCluster(getOption("cl.cores", mc.cores))

  res <- parallel::parLapply(cl = cl, X = X, fun = FUN, ...)

  parallel::stopCluster(cl)

  unregister_dopar()

  return(res)

}

mclapply <- function(X, FUN, ..., mc.cores){

  cores <- parallel::detectCores()

  cl <- min(mc.cores, cores - 1)

  doParallel::registerDoParallel(cl)

  if(Sys.info()['sysname'] == 'Windows'){

    res <- mclapply.win(X = X, mc.cores = cl, FUN = FUN, ...)

  }else{

    res <- parallel::mclapply(X = X, mc.cores = cl, FUN = FUN, ...)

  }

  return(res)

}


#Top variable####

topvarfeatures <- function(betasmat,
                           topfeaturenumber = NULL){

  limit <- min(ncol(betasmat), topfeaturenumber)
  betas.filtered <- betasmat[,order(apply(betasmat, 2, sd), decreasing = TRUE)[1:limit]]

  if(ncol(betas.filtered) < 10){
    sigmat <- betasmat
  }else{
    sigmat <- betas.filtered
  }

  return(sigmat)

}


#CV####

#Add parameter seednum to the original function `makecv` and its subfunction
#`makefolds`
makefolds <- function(y, cv.fold = 5, seednum = 1234){
  n <- length(y)
  nlvl <- table(y)
  idx <- numeric(n)
  folds <- list()
  for (i in 1:length(nlvl)) {
    set.seed(seednum)
    idx[which(y == levels(y)[i])] <- sample(rep(1:cv.fold,length = nlvl[i]))
  }
  for (i in 1:cv.fold){
    folds[[i]] <- list(train = which(idx!=i),
                       test =  which(idx==i))
  }
  return(folds)
}



#'Divide samples into different cross validation loops
#'
#'Create a cross validation structure and assign samples to different loops
#'
#'@param y The true labels of the samples. Can be a vector or factor. Each of
#'  its element is a label for a sample and the element order in it should be
#'  the same as the sample order in the sample data provided to the function
#'  \code{maincv} to do cross validation for classification models.
#'@param cv.fold A numeric number and the default value is 5, so that if the
#'  parameter \code{normalcv} is TRUE, a 5 fold cross validation structure
#'  will be created, and if \code{normalcv} is FALSE, a 5 by 5 nested cross
#'  validation structure will be created.
#'@param seednum The random seed for the sampling process to assign samples to
#'  different cross validation loops. Default is 1234.
#'@param normalcv Indicating whether the cross validation loops are normal or
#'  nested. Default is FALSE, meaning nested cross validation.
#'@return Will return a multiple layer list recording the samples assigned to
#'  each cross validation loop and the samples are indicated by their indeces
#'  in \code{y}.
#'@export
makecv <- function(y,
                   cv.fold = 5,
                   seednum = 1234,
                   normalcv = FALSE){

  if(!is.factor(y)){
    y <- as.character(y)
    freqy <- table(y)
    freqy <- freqy[order(-freqy)]
    y <- factor(y, levels = names(freqy), ordered = TRUE)

  }

  nfolds <- list()
  folds <- makefolds(y, cv.fold, seednum = seednum)
  names(folds) <- paste0("outer",1:length(folds))

  for(k in 1:length(folds)){

    if(normalcv == FALSE){

      inner = makefolds(y[folds[[k]]$train],
                        cv.fold,
                        seednum = seednum)

      names(inner) <- paste0("inner",1:length(folds))

      for(i in 1:length(inner)){
        inner[[i]]$train <- folds[[k]]$train[inner[[i]]$train]
        inner[[i]]$test <- folds[[k]]$train[inner[[i]]$test]
      }

      nfolds[[k]] <- list(folds[k],inner)

    }else{

      nfolds[[k]] <- list(folds[k])

    }
  }

  names(nfolds) <- paste0("outer",1:length(nfolds))

  return(nfolds)

}


#Generate CV beta data


#'Generate the data matrices for the cross validation loops
#'
#'Generate the training and testing data matrices for all the cross validation
#'loops.
#'
#'@param y The true labels of the samples. Can be a vector or factor. Each of
#'  its element is a label for a sample and the element order in it should be
#'  the same as the sample order in the data matrix provided to this function
#'  via the parameter \code{betas}. If the parameter \code{nfolds..} is not
#'  NULL, this parameter is not necessary.
#'@param betas The beta value matrix of the samples. Each row is one sample
#'  and each column is one feature. It can also be set as NULL, and in this
#'  case, the function will read the data by the directory \code{fpath.betas}.
#'  The absolute path of the file of these data should be provided by the
#'  parameter \code{fpath.betas}.
#'@param fpath.betas If the parameter \code{betas} is NULL, this parameter is
#'  necessary to provide the file path of the betas matrix data, so that the
#'  function will read the data from this path. It should be an absolute path
#'  string, and the file should be an .rds file.
#'@param topfeaturenumber It is used to set the prescreened feature number so
#'  that before the training/testing sets division, the data features will be
#'  reduced to only the top variable ones and the number of these retained top
#'  variable features is \code{topfeaturenumber}. Default is 50000. It can be
#'  set as NULL too, so that no precreen will be done on the data.
#'@param n.cv.folds A numeric number and the default value is 5, so that if
#'  the parameter \code{normalcv} is set as TRUE, a 5 fold cross validation
#'  will be set up, and if \code{normalcv} is FALSE, a 5 by 5 nested cross
#'  validation will be set up. This parameter only works if \code{y} is not
#'  NULL.
#'@param seed The random seed for the sampling process to assign samples to
#'  different cross validation loops. Default is 1234.
#'@param normalcv Indicating whether the cross validation loops are normal or
#'  nested. Default is FALSE, meaning nested cross validation.
#'@param nfolds.. If the parameter \code{n.cv.folds} or \code{y} is NULL, the
#'  training/testing sets need to be divided following the cross validation
#'  structure provided by this parameter and it is the result of the function
#'  \code{makecv}, but if it is also NULL, and there is such a data structure
#'  named "nfolds" in the environment, it will be loaded by the function and
#'  use it to divide the training/testing sets.
#'@param out.path For all the cross validation loops, their beta value matrix
#'  files will be saved in the folder set by this parameter. It is the folder
#'  name will be created in the current working directory, so a relative not
#'  absolute path.
#'@param out.fname.prefix The beta value matrix files of the cross validation
#'  loops will be saved in the folder set by the parameter \code{out.path} and
#'  the name prefix of the files need to be set by this parameter.
#'@return The beta value matrix files for the cross validation loops will be
#'  saved in the directory set by \code{out.path}, and each cross validation
#'  loop will have an .RData file saved there, with the beta value matrices
#'  for the training set and testing set, and the list indicating the training
#'  and testing sample indexes of that loop in it. The absolute directory of
#'  the folder can be transferred to the function \code{maincv} to construct
#'  and evaluate the model performance via cross validation.
#'@export
cvdata <- function(y = NULL,
                   betas = NULL,
                   fpath.betas,

                   topfeaturenumber = 50000,

                   n.cv.folds = 5,
                   seed = 1234,
                   normalcv = FALSE,
                   nfolds.. = NULL,

                   out.path = 'betas.train.test.filtered',
                   out.fname.prefix = 'betas'

){

  K.start <- 1
  k.start <- 0

  if(!is.null(y)){
    if(!is.factor(y)){
      y <- as.character(y)
      freqy <- table(y)
      freqy <- freqy[order(-freqy)]
      y <- factor(y, levels = names(freqy), ordered = TRUE)
    }
  }

  if(!is.null(n.cv.folds) && !is.null(y)){

    nfolds.. <- makecv(y = y,
                       cv.fold = n.cv.folds,
                       seednum = seed,
                       normalcv = normalcv)

  }else if(is.null(nfolds..) && exists("nfolds")){

    nfolds.. <- get("nfolds", envir = .GlobalEnv)
    n.cv.folds <- length(nfolds..)

  }

  if(!is.null(betas)){

    betasv11.h5 <- betas

  }else if(!is.null(fpath.betas)){

    basepath <- unlist(strsplit(x = fpath.betas, split = '/', fixed = TRUE))
    basepath <- paste(basepath[-length(basepath)], collapse = '/')

    fpath.betas <- file.path(fpath.betas)

    betasv11.h5 <- readRDS(fpath.betas)

  }


  if(!is.null(topfeaturenumber)){
    betasv11.h5 <- topvarfeatures(betasmat = betasv11.h5,
                                  topfeaturenumber = topfeaturenumber)

  }

  betasv11.h5 <- t(betasv11.h5)
  subset.CpGs <- nrow(betasv11.h5)


  #Run CV scheme
  message("Cross validation (CV) scheme starts ... @ ", Sys.time(), "\n")

  for(K in K.start:n.cv.folds){

    for(k in k.start:n.cv.folds){
      cat(paste0('k = ', k, '\n'))

      if(k > 0 & normalcv == FALSE){

        message("Calculating inner/nested fold ", K,".", k,"  ... @ ",Sys.time(), "\n")
        fold <- nfolds..[[K]][[2]][[k]]

      }else if(k > 0 & normalcv == TRUE){

        next()

      }else{

        message("Calculating outer fold ", K,".0  ... @ ",Sys.time(), "\n")
        fold <- nfolds..[[K]][[1]][[1]]

      }

      # Subset K.k$train
      message("Subsetting cases/columns: " , K, ".", k, " training set @ ", Sys.time(), "\n")

      betas.K.k.train <- betasv11.h5[, fold$train]

      if(!is.null(subset.CpGs)){

        betas.p.filtered.K.k.train <- betas.K.k.train[order(apply(betas.K.k.train, 1, sd),
                                                            decreasing = TRUE)[1:subset.CpGs], ]

      }else{
        betas.p.filtered.K.k.train <- betas.K.k.train
      }

      message("Check whether there is NA in train set: ",
              sum(is.na(betas.p.filtered.K.k.train) == TRUE), "\n")

      # Transposed afterwards!
      betas.p.filtered.K.k.train <- t(betas.p.filtered.K.k.train)

      # Garbage collector (note: gc is not absolutely necessary)
      message("Clean up memory (garbage collector) @ ", Sys.time(), "\n")
      gc()

      message("Subsetting " , K, ".", k, " test/calibration set @ ", Sys.time(), "\n")
      betas.K.k.test <- betasv11.h5[, fold$test]

      betas.p.filtered.K.k.test <- betas.K.k.test[match(colnames(betas.p.filtered.K.k.train),
                                                        rownames(betas.K.k.test)), ]

      betas.p.filtered.K.k.test <- t(betas.p.filtered.K.k.test)

      betas.K.k <- betasv11.h5[match(colnames(betas.p.filtered.K.k.train), rownames(betasv11.h5)), ]

      betas.K.k <- t(betas.K.k)

      folder.path <- file.path(getwd(), "data", out.path)

      if(!dir.exists(folder.path)){
        dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
      }

      betas.train <- betas.p.filtered.K.k.train
      rm(betas.p.filtered.K.k.train)
      betas.test <- betas.p.filtered.K.k.test
      rm(betas.p.filtered.K.k.test)

      save(betas.train,
           betas.test,
           fold,
           file = file.path(folder.path, paste(out.fname.prefix, K, k, "RData", sep = "."))
      )

    }
  }

}

subfunc_nestedcv_scheduler <- function(K, K.start, K.stop, k.start, k.stop, n.cv.folds, n.cv.inner.folds){

  # Correctly stop nested CV
  if(K > K.start && K < K.stop) {
    k.start <- 0
    n.cv.inner.folds <- n.cv.folds

  } else {
    if(K == K.start) {
      if(K.start != K.stop) { # && k.start != 0){
        n.cv.inner.folds <- n.cv.folds # k inner goes to .5
      } else { # K == K.start == K.stop && k.start == 0
        n.cv.inner.folds <- k.stop
      }
    } else { # K == K.stop
      if(k.start != 0) k.start <- 0
      n.cv.inner.folds <- k.stop
    }
  }
  res <- list(k.start = k.start,
              n.cv.inner.folds = n.cv.inner.folds
  )
  return(res)
}

#SCMER####

scmerselection <- function(trimbetasmat,
                           annodat = NULL,
                           labeldat = NULL,
                           K = NULL,
                           k = NULL,
                           lasso = 5.5e-6,
                           ridge = 0,
                           n_pcs = 100,
                           perplexity = 30,
                           threads = 6,
                           savefigures = FALSE,
                           pythonpath = NULL,
                           scmerpyfile){

  if(!is.null(pythonpath)){

    Sys.setenv(RETICULATE_PYTHON = pythonpath)

  }

  #reticulate::use_python(pydir)

  reticulate::py_config()

  reticulate::source_python(scmerpyfile)

  probes <- colnames(trimbetasmat)
  samples <- row.names(trimbetasmat)

  if(is.null(K)){

    K <- Sys.Date()

  }

  if(is.null(k)){

    k <- format(Sys.time(), '%T')
    k <- gsub(pattern = ':', replacement = '-', x = k)

  }

  K <- as.character(K)
  k <- as.character(k)

  if(is.null(annodat) | is.null(labeldat)){
    annodat <- NULL
    labeldat <- NULL
  }

  scmerfeatrues <- scmerpy(trimbetasmat = trimbetasmat,
                           probes = probes,
                           samples = samples,
                           annodat = annodat,
                           labeldat = labeldat,
                           K = K,
                           k = k,
                           lasso = lasso,
                           ridge = ridge,
                           n_pcs = n_pcs,
                           perplexity = perplexity,
                           threads = threads,
                           savefigures = savefigures)

  scmerfeatrues <- scmerfeatrues$features

  return(scmerfeatrues)

}


#PreScreen#####

prescreenfeatures <- function(betasmat,
                              labeldat,
                              cutoff = 0.01,
                              threads = 6,
                              topfeaturenumber = NULL){

  interceptmod <- nnet::multinom(labeldat ~ 1)

  featurepval <- function(featuredat = betasmat,
                          featureidx,
                          response = labeldat,
                          intermod = interceptmod){

    featuremod <- nnet::multinom(response ~ featuredat[,featureidx])

    #Likelihood ratio test
    lrres <- lmtest::lrtest(featuremod, intermod)
    lrpval <- lrres$`Pr(>Chisq)`
    lrpval <- lrpval[-1]

    return(lrpval)

  }

  seqs <- seq(1, ncol(betasmat), 1)
  featurepvals <- mclapply(X = seqs,
                           FUN = featurepval,  featuredat = betasmat,
                           response = labeldat, intermod = interceptmod,
                           mc.cores = threads)
  featurepvals <- unlist(featurepvals)
  #featurepvals <- p.adjust(p = featurepvals, method = c('BH'))
  names(featurepvals) <- colnames(betasmat)
  featurepvals <- featurepvals[!is.na(featurepvals)]


  sigfeaturepvals <- featurepvals[featurepvals < cutoff]
  sigfeaturepvals <- sigfeaturepvals[order(as.vector(sigfeaturepvals))]
  sigfeaturepvals <- sigfeaturepvals[1:min(length(sigfeaturepvals), topfeaturenumber)]

  filetag <- Sys.time()
  filetag <- gsub(pattern = ':', replacement = '-', x = filetag)
  filetag <- gsub(pattern = ' ', replacement = '-', x = filetag)
  save(sigfeaturepvals, file = paste0('sigfeaturepvals', '.', filetag, '.RData'))

  sigfeatures <- names(sigfeaturepvals)

  if(length(sigfeatures) < 10){
    sigmat <- betasmat
  }else{
    sigmat <- betasmat[,sigfeatures]
  }


  return(sigmat)

}



#RF####

#Parallelized randomForest wrapper

rfp <- function(xx, ..., ntree = ntree, mc = mc, seed = 1234){

  rfwrap <- function(ntree, xx, ...){
    set.seed(seed)
    randomForest::randomForest(x = xx, ntree = ntree, norm.votes = FALSE, ...)
  }

  rfpar <- mclapply(rep(ceiling(ntree / mc), mc), mc.cores = mc, rfwrap, xx = xx, ...)

  do.call(randomForest::combine, rfpar)

}

#Training & tuning function

trainRF <- function(y, betas, ntrees, p, seed, cores){

  set.seed(seed)
  rf.varsel <- rfp(xx = betas,
                   y,
                   mc = cores,
                   ntree = ntrees,
                   strata = y,
                   sampsize = rep(min(table(y)),length(table(y))),
                   importance = TRUE,
                   replace = FALSE,
                   seed = seed)


  # extract permutation based importance measure
  imp.perm <- randomForest::importance(rf.varsel, type = 1)

  # variable selection
  or <- order(imp.perm, decreasing = T)
  betasy <- betas[ , or[1:p]]   # CAVE: p (argument in trainRF) => limits the number of most important variable (importance.perm)
  # betasy = only 100! (p = 100 in MNPrandomForest.R)

  set.seed(seed)
  rf.pred <- randomForest::randomForest(betasy,
                                        y,
                                        ntree = ntrees,
                                        strata = y,
                                        sampsize = rep(min(table(y)), length(table(y))),
                                        proximity = TRUE,
                                        oob.prox = TRUE,
                                        importance = TRUE,
                                        keep.inbag = TRUE,
                                        do.trace = FALSE)

  res <- list(rf.pred, imp.perm)

  return(res)
}

#Calibration####

subfunc_Platt_train_calibration  <- function(y.i.j.innertest,
                                             scores.i.j.innertest,
                                             diagn.class){
  y.calib.true.diagn01 <- ifelse(y.i.j.innertest == diagn.class, 1, 0) # 1-vs-all
  y.calib.pred.diagn <- scores.i.j.innertest[ , colnames(scores.i.j.innertest) == diagn.class] # class2show
  calib.df <- data.frame(cbind(y.calib.true.diagn01, y.calib.pred.diagn)) # # slow step DF are rewritten in the memory; list are not <-  AdvancedR H.Wickham.
  colnames(calib.df) <- c("y", "x")
  calib.model.Platt.diagn.i <- glm(y ~ x, calib.df, family=binomial)
  return(calib.model.Platt.diagn.i)
}


subfunc_Platt_fit_testset <- function(scores.i.0.outertest,
                                      calib.model.Platt.diagn.i,
                                      diagn.class){
  y.test.pred.diagn <- scores.i.0.outertest[ , colnames(scores.i.0.outertest) == diagn.class] # CAVE: here scores <= OUTER FOLD$TEST!
  test.df <- data.frame(y.test.pred.diagn) # slow step DFs are rewritten in the memory;
  colnames(test.df) <- c("x")
  probs.platt.diagn.i <- predict(calib.model.Platt.diagn.i, newdata = test.df, type = "response")
  return(probs.platt.diagn.i)
}


subfunc_Platt_train_calibration_Firth  <- function(y.i.j.innertest,
                                                   scores.i.j.innertest,
                                                   diagn.class,
                                                   brglm.control.max.iteration){
  y.calib.true.diagn01 <- ifelse(y.i.j.innertest == diagn.class, 1, 0)
  y.calib.pred.diagn <- scores.i.j.innertest[ , colnames(scores.i.j.innertest) == diagn.class] # class2show
  calib.df <- data.frame(cbind(y.calib.true.diagn01, y.calib.pred.diagn)) # slow step DF are rewritten in the memory
  colnames(calib.df) <- c("y", "x")
  calib.model.Platt.Firth.diagn.i <- brglm::brglm(formula = y ~ x, data = calib.df, family = binomial(logit),
                                                  method = "brglm.fit",
                                                  control.brglm = brglm::brglm.control(br.maxit = brglm.control.max.iteration)) # default is br.maxit = 100!
  # further controls left at default: br.epsilon = 1e-08 ;
  return(calib.model.Platt.Firth.diagn.i)
}

subfunc_Platt_fit_testset_Firth <- function(scores.i.0.outertest,
                                            calib.model.Platt.Firth.diagn.i,
                                            diagn.class){
  y.test.pred.diagn <- scores.i.0.outertest[ , colnames(scores.i.0.outertest) == diagn.class] # CAVE here scores = outer fold $test
  test.df <- data.frame(y.test.pred.diagn) # slow step DF are rewritten in the memory
  colnames(test.df) <- c("x")
  probs.platt.firth.diagn.i <- predict(calib.model.Platt.Firth.diagn.i, newdata = test.df, type = "response")
  return(probs.platt.firth.diagn.i)
}

#Evaluation#####

brier <- function(scores,y){
  ot <- matrix(0,nrow=nrow(scores),ncol=ncol(scores))
  # It can cause error:
  # encountered errors in user code, all values of the jobs will be affectedError in matrix(0, nrow = nrow(scores), ncol = ncol(scores))
  arr.ind <- cbind(1:nrow(scores),match(y,colnames(scores)))
  ot[arr.ind] <- 1
  sum((scores - ot)^2)/nrow(scores)
}

mlogloss <- function(scores,y){
  N <- nrow(scores)
  y_true <- matrix(0,nrow=nrow(scores),ncol=ncol(scores))
  arr.ind <- cbind(1:nrow(scores),match(y,colnames(scores)))
  y_true[arr.ind] <- 1
  eps <- 1e-15 # we use Kaggle's definition of multiclass log loss with this constrain (eps) on extremly marginal scores (see reference below)
  scores <- pmax(pmin(scores, 1 - eps), eps)
  (-1 / N) * sum(y_true * log(scores))
}

subfunc_misclassification_rate <- function(y.true.class, y.predicted){
  error_misclass <- sum(y.true.class != y.predicted)/length(y.true.class)
  return(error_misclass)
}

subfunc_multiclass_AUC_HandTill2001 <- function(y.true.class, y.pred.matrix.rowsum.scaled1){
  auc_multiclass <- HandTill2001::auc(HandTill2001::multcap(response = factor(x = y.true.class,
                                                                              levels = colnames(y.pred.matrix.rowsum.scaled1)),
                                                            predicted = y.pred.matrix.rowsum.scaled1))
  return(auc_multiclass)
}


#SVM-LK####
## Cost tuner subfunction

subfunc_svm_e1071_linear_train_tuner_mc <- function(data.xTrain,
                                                    target.yTrain,
                                                    mod.type = "C-classification",
                                                    kernel. = "linear",
                                                    scale. = T,
                                                    C.base = 10,
                                                    C.min = -3,
                                                    C.max = 3,
                                                    n.CV = 5,
                                                    verbose = T,
                                                    seed = 1234,
                                                    parallel = T,
                                                    mc.cores = 4L,
                                                    weighted = FALSE){

  # Cost C grid + give feedback and Sys.time
  Cost.l <- as.list(C.base^(C.min:C.max))
  message("\nCost (C) = ", paste(simplify2array(Cost.l), sep = " ", collapse = " ; "),
          " ; \nNr. of iterations: ", length(Cost.l),
          "\nStart at ", Sys.time())
  # Predefine empty list for results
  cvfit.e1071.linear.C.tuner <- list()

  basesvm <- function(i,
                      seed,
                      data.xTrain,
                      target.yTrain,
                      scale.,
                      mod.type,
                      kernel.,
                      Cost.l,
                      n.CV,
                      weighted = FALSE){

    if(weighted == TRUE){

      classweights <- 100/table(target.yTrain)

      set.seed(seed + 1, kind ="default")

      res <- e1071::svm(x = data.xTrain,
                        y = target.yTrain,
                        scale = scale.,
                        type = mod.type,
                        kernel = kernel.,
                        cross = n.CV,
                        probability = TRUE,
                        fitted = TRUE,

                        class.weights = classweights)

    }else{
      classweights <- NULL

      set.seed(seed + 1, kind ="default")

      res <- e1071::svm(x = data.xTrain,
                        y = target.yTrain,
                        scale = scale.,
                        type = mod.type,
                        kernel = kernel.,
                        cost = Cost.l[[i]],
                        cross = n.CV,
                        probability = TRUE,
                        fitted = TRUE,

                        class.weights = classweights)

    }

    return(res)

  }

  # Parallel
  if(parallel){
    cvfit.e1071.linear.C.tuner <- mclapply(X = seq_along(Cost.l),
                                           FUN = basesvm,
                                           seed = seed,
                                           data.xTrain = data.xTrain,
                                           target.yTrain = target.yTrain,
                                           scale. = scale.,
                                           mod.type = mod.type,
                                           kernel. = kernel.,
                                           Cost.l = Cost.l,
                                           n.CV = n.CV,
                                           weighted = weighted,
                                           #mc.preschedule = T,
                                           #mc.set.seed = T,
                                           mc.cores = mc.cores)
    print(Sys.time())
    return(cvfit.e1071.linear.C.tuner)

    # Sequential
  } else {
    cvfit.e1071.linear.C.tuner <- lapply(seq_along(Cost.l),
                                         FUN = basesvm,
                                         seed = seed,
                                         data.xTrain = data.xTrain,
                                         target.yTrain = target.yTrain,
                                         scale. = scale.,
                                         mod.type = mod.type,
                                         kernel. = kernel.,
                                         Cost.l = Cost.l,
                                         n.CV = n.CV,
                                         weighted = weighted)
    print(Sys.time())
    return(cvfit.e1071.linear.C.tuner)
  }
}

## Cost selector subfunction
subfunc_svm_e1071_linear_C_selector <- function(results.cvfit.e1071.linear.C.tuner,
                                                C.base = 10, C.min = -3, C.max = 3,
                                                n.CV = 5, verbose = T){

  Costs.l <- as.list(C.base^(C.min:C.max))
  # Print simplified version of each crossvalidated fold accuracy for eventual manual selection
  res.cvfit.svm.accuracies.nCV <- sapply(seq_along(C.base^(C.min:C.max)), function(i){
    simplify2array(results.cvfit.e1071.linear.C.tuner[[i]]$accuracies)})
  colnames(res.cvfit.svm.accuracies.nCV) <- paste0("Cost_", Costs.l)
  rownames(res.cvfit.svm.accuracies.nCV) <- paste0("nCV", seq(1, n.CV, 1))
  # Print matrix of all CV accuracies
  if(verbose){
    message("\nMatrix of all CV accuracies:")
    print(res.cvfit.svm.accuracies.nCV)
  }

  # Average accuracy
  res.cvfit.svm.accuracies.mean <- sapply(seq_along(C.base^(C.min:C.max)), function(i){
    simplify2array(results.cvfit.e1071.linear.C.tuner[[i]]$tot.accuracy)})
  names(res.cvfit.svm.accuracies.mean) <- paste0("Cost_", Costs.l)
  # Same as: res.cvfit.svm.accuracies.mean <- apply(res.cvfit.svm.accuracies.nCV, 2, mean)
  # Print list of average CV accuracies/ $tot.accuracy
  if(verbose){
    message("\nMean CV accuracies:")
    print(res.cvfit.svm.accuracies.mean)
  }

  # Selection
  # Chooses the smallest C with highest 5-fold cross-validated accuracy among possible choices
  # => if C is large enough anyway (see Appendix-N.5.) doesnt make a difference
  # => saves also computation time if C is smaller # => error-margin/Nr of supp.vecs.
  C.selected <- Costs.l[[which.max(res.cvfit.svm.accuracies.mean)]]
  message("\nCost parameter with highest ", n.CV, "-fold CV accuracy : C = ", C.selected, " ; ",
          "\n Note: If more than one maximal accuracy exists, C returns the smallest cost parameter with highest accuracy.",
          "\n Once C is large than a certain value, the obtained models have similar performances",
          " (for theoretical proof see Theorem 3 of Keerthi and Lin, 2003)")
  res <- list(C.selected = C.selected,
              mtx.accuracies.nCV = res.cvfit.svm.accuracies.nCV,
              mtx.accuracies.mean = res.cvfit.svm.accuracies.mean)
  return(res)

  # Literature:
  # Important # A practical guide to LIBLINEAR - Fan, Chang, Hsiesh, Wang and Lin 2008
  # <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>
  # Appendix-N.5. Parameter Selection:
  # 1. Solvers in LIBLINEAR are not very sensitive to C. Once C is large than a certain value,
  # the obtained models have similar performances.
  # Theoretical proof: Theorem 3 of Keerthi and Lin (2003)
}

## Re-fit training data subfunction

subfunc_svm_e1071_linear_modfit_train <- function(C.tuned,
                                                  data.xTrain, target.yTrain,
                                                  results.cvfit.e1071.linear.C.tuner,
                                                  C.selector.accuracy.mean,
                                                  use.fitted = T){  #res.svm.C.tuner.l

  message("\n\nRe-fitting training data ... ", Sys.time())
  #Costs.l <- as.list(C.base^(C.min:C.max))
  i <- which.max(C.selector.accuracy.mean)

  # Ver.1 - Use predict function to refit training data
  # Note If the training set was scaled by svm (done by default),
  # the new data is scaled accordingly using scale and center of the training data.
  modfit.train.svm.lin.pred <- tryCatch({

    predict(object = results.cvfit.e1071.linear.C.tuner[[i]],
            newdata =  data.xTrain,
            decision.values = T,
            # decision values of all binary classif. in multiclass setting are returned.
            probability = T)

  }, error = function(err){

    e1071:::predict.svm(object = results.cvfit.e1071.linear.C.tuner[[i]],
                        newdata =  data.xTrain,
                        decision.values = T,
                        # decision values of all binary classif. in multiclass setting are returned.
                        probability = T)

  })

  # Ver.2 - Use fitted() - see ??svm - Examples
  if(use.fitted){modfit.train.svm.lin.fitted <- fitted(results.cvfit.e1071.linear.C.tuner[[i]])}
  #message("\nBoth predict() & fitted() are ready @ ", Sys.time())
  message("\nPrediction is ready @ ", Sys.time())

  # Output
  res <- list(svm.e1071.model.object = results.cvfit.e1071.linear.C.tuner[[i]],
              trainfit.svm.lin1 = modfit.train.svm.lin.pred,
              trainfit.svm.lin2 = modfit.train.svm.lin.fitted) # => output file is rel. large / lot of large mtx.
  return(res)
}


### Training & tuning function

train_SVM_e1071_LK <- function(y, betas.Train,
                               seed,
                               mc.cores,
                               nfolds = 5,
                               C.base = 10, C.min = -3, C.max = 3,
                               scale.internally.by.e1071.svm = T,
                               mod.type = "C-classification",
                               weighted = FALSE){

  ## 1. Crossvalidate SVM/LiblineaR - Cost parameter for optimal
  set.seed(seed, kind = "default")

  message("\nTuning SVM (e1071) linear kernel: hyperparameter C (cost) ... ", Sys.time())

  cvfit.svm.e1071.linear.C.tuner <- subfunc_svm_e1071_linear_train_tuner_mc(data.xTrain = betas.Train,
                                                                            target.yTrain = y,
                                                                            mod.type = mod.type,
                                                                            kernel. = "linear",
                                                                            scale. = scale.internally.by.e1071.svm,
                                                                            C.base = C.base,
                                                                            C.min = C.min, C.max = C.max,
                                                                            n.CV = 5,
                                                                            verbose = T,
                                                                            seed = seed,
                                                                            parallel = T,
                                                                            mc.cores = mc.cores,
                                                                            weighted = weighted)


  # Extract optimal C or smallest C with highest accuracy
  C.tuned.cv <-  subfunc_svm_e1071_linear_C_selector(results.cvfit.e1071.linear.C.tuner = cvfit.svm.e1071.linear.C.tuner,
                                                     C.base = C.base,
                                                     C.min = C.min,
                                                     C.max = C.max,
                                                     n.CV = nfolds,
                                                     verbose = T)


  # C.tuned.cv = list of 3: $C.selected $mtx.accuracies.nCV $mtx.accuracies.mean
  # Provide message with value
  message(paste0("Optimal cost (C) parameter: ", C.tuned.cv$C.selected))

  # Refit models on s.xTrain L2R_LR (type0 - ca. 15 min/refit) and optionally only for classes Crammer & Singer (type4 - it takes just ca. +35s)
  message("\n(Re)Fitting optimal/tuned model on training data ... ", Sys.time())

  modfit.svm.linear.train <- subfunc_svm_e1071_linear_modfit_train(C.tuned = C.tuned.cv$C.selected,
                                                                   data.xTrain = betas.Train,
                                                                   target.yTrain = y,
                                                                   results.cvfit.e1071.linear.C.tuner = cvfit.svm.e1071.linear.C.tuner,
                                                                   C.selector.accuracy.mean = C.tuned.cv$mtx.accuracies.mean,
                                                                   use.fitted = T)
  # uses predict supposed to scale data.xTrain / betas.Train automatically


  # CAVE conames order is not the same as in levels(y) !!!
  pred.scores.trainfit.svm.lin1 <- attr(modfit.svm.linear.train$trainfit.svm.lin1, "probabilities")

  # Results
  res <- list(modfit.svm.linear.train$svm.e1071.model.object,
              modfit.svm.linear.train$trainfit.svm.lin1,
              pred.scores.trainfit.svm.lin1,
              modfit.svm.linear.train$trainfit.svm.lin2,
              cvfit.svm.e1071.linear.C.tuner,
              C.tuned.cv)
  return(res)
}


#XGBoost####
### Define training & tuning function

trainXGBOOST_caret_tuner <- function(y,
                                     train.K.k.mtx,
                                     K., k.,
                                     dtrain., watchlist.,
                                     n.CV = 3, n.rep = 1,
                                     seed.,
                                     allow.parallel = T,
                                     mc.cores = 4L,
                                     max_depth. = 6,
                                     eta. = c(0.1, 0.3),
                                     gamma. = c(0, 0.01),
                                     colsample_bytree. = c(0.01, 0.02, 0.05, 0.2),
                                     minchwght = 1,
                                     subsample. = 1,
                                     nrounds. = 100,
                                     early_stopping_rounds. = 50,
                                     objective. = "multi:softprob",
                                     eval_metric. = "merror",
                                     save.xgb.model = T,
                                     out.path.model.folder = "xgboost-train-best-model-object",
                                     save.xgb.model.name = "xgboost.model.train.caret"){

  ## 1.
  set.seed(seed., kind = "default")
  message("seed: ", seed.)
  message("n: ", nrow(dtrain.))

  # Security check for nested multicores/threads
  colsamp.l <- as.list(colsample_bytree.)

  # CARET - Grid
  xgbGrid <- expand.grid(nrounds = nrounds.,
                         max_depth = max_depth.,
                         eta = eta.,
                         gamma = gamma.,
                         colsample_bytree = colsample_bytree.,
                         min_child_weight = minchwght,
                         subsample = subsample.)

  message("\nTuning grid size: ", nrow(xgbGrid))

  # CARET - trControl
  xgbTrControl <- caret::trainControl(
    method = "repeatedcv",
    number = n.CV,
    repeats = n.rep,
    verboseIter = TRUE,
    returnData = FALSE,
    classProbs = F,
    summaryFunction = caret::multiClassSummary,
    allowParallel = allow.parallel
  )

  # CARET - train
  Sys.time()
  set.seed(seed = seed., kind = "default")
  xgb.Train.caret.res <- caret::train(x = train.K.k.mtx,
                                      y = y,
                                      method = "xgbTree",
                                      trControl = xgbTrControl,
                                      tuneGrid = xgbGrid
  )

  message("\nResults of caret grid search:")
  print(xgb.Train.caret.res)

  # Rerun best model => Switch to xgboost - xgb.train
  param.xgb.train <- list(max_depth = xgb.Train.caret.res$bestTune$max_depth,
                          eta = xgb.Train.caret.res$bestTune$eta,
                          gamma = xgb.Train.caret.res$bestTune$gamma,
                          colsample_bytree = xgb.Train.caret.res$bestTune$colsample_bytree,
                          min_child_weight = xgb.Train.caret.res$bestTune$min_child_weight, # controls the hessian
                          subsample = xgb.Train.caret.res$bestTune$subsample,
                          num_class = length(levels(y)), # 91
                          objective = "multi:softprob",
                          eval_metric = "merror",
                          nthread = mc.cores)   #allow all cores!

  # Create output directory - for saving xgb.train model objects for eventual later loading/use
  folder.path <- file.path(getwd(), out.path.model.folder)
  dir.create(folder.path, recursive = TRUE, showWarnings = FALSE)
  # Saving path and object name
  save.xgb.model.train.name <- file.path(folder.path, paste(save.xgb.model.name, "train.autosave", K., k., sep = "."))

  # Re-Run xgb.train to find optimal niter/nrounds & concurrently check test error => get Model object
  message("Re-Run xgb.train to find optimal niter/nrounds & concurrently check test error & save model object @ ", Sys.time())
  xgb.Train.model.caret.best <-  xgboost::xgb.train(params = param.xgb.train,
                                                    data = dtrain.,
                                                    nrounds = xgb.Train.caret.res$bestTune$nrounds,
                                                    watchlist = watchlist.,
                                                    verbose = 1, print_every_n = 1L,
                                                    early_stopping_rounds = early_stopping_rounds.,
                                                    maximize = F,
                                                    save_period = 0,
                                                    save_name = save.xgb.model.train.name,
                                                    xgb_model = NULL)
  message("\nBest {caret} CV model - performance using `xgb.train()`: ")
  print(xgb.Train.model.caret.best, verbose = T)

  message("\n(Re)Fitting tuned model with optimal nrounds/ntreelimit on training data ... ", Sys.time())
  scores.pred.xgboost.vec.train <- predict(object = xgb.Train.model.caret.best,
                                           newdata = dtrain.,
                                           ntreelimit = xgb.Train.model.caret.best$best_iteration,
                                           outputmargin = F)

  # Generate a matrix from the vector of probabilities
  scores.xgboost.tuned.train <- matrix(scores.pred.xgboost.vec.train, nrow = nrow(dtrain.),
                                       ncol = length(levels(y)), byrow = T)
  # Reassign row & colnames
  rownames(scores.xgboost.tuned.train) <- rownames(train.K.k.mtx)
  colnames(scores.xgboost.tuned.train) <- levels(y)

  # Results
  res <- list(xgb.Train.model.caret.best,
              scores.xgboost.tuned.train,
              xgb.Train.caret.res)

  return(res)
}


#GLMNET####

subfunc_cvglmnet_tuner_mc_v2 <- function(x, # needs to be a matrix object # df gives error!
                                         y,
                                         family = "multinomial",
                                         type.meas = "mse",
                                         alpha.min = 0, alpha.max = 1, by = 0.1,
                                         n.lambda = 100,
                                         lambda.min.ratio = 10^-6,
                                         nfolds = 10,
                                         balanced.foldIDs = T,
                                         seed = 1234,
                                         parallel.comp = T,
                                         mc.cores=2L){

  alpha.grid <- as.list(seq(alpha.min, alpha.max, by))

  message("Setting balanced foldIDs with nfold = ", nfolds, " @ ", Sys.time())
  set.seed(seed)
  if(balanced.foldIDs){foldIDs <- c060::balancedFolds(class.column.factor = y,
                                                      cross.outer = nfolds)}

  # Define new version/repurposed `cv.glmnet()` function parellized for the alpha grid
  parallel.cvglmnet <- function(i){
    message("Tuning cv.glmnet with alpha = ", i, " @ ", Sys.time())
    set.seed(seed + 1, kind ="default")
    glmnet::cv.glmnet(x = x,
                      y = y,
                      alpha = i,
                      family = family,
                      type.measure = type.meas,
                      nlambda = n.lambda,
                      lambda.min.ratio = lambda.min.ratio,
                      foldid = foldIDs,
                      parallel = parallel.comp)
  }

  message("Start mclapply ", " @ ", Sys.time())
  cvfit.l <- mclapply(alpha.grid, parallel.cvglmnet,
                      mc.cores = mc.cores)


  return(cvfit.l)
}


subfunc_glmnet_mse_min_alpha_extractor <- function(resl,
                                                   lambda.1se = T,
                                                   alpha.min = 0,
                                                   alpha.max = 1){
  outl <- list(length(resl))
  if(lambda.1se){
    outl <- sapply(seq_along(resl), function(i){
      resl[[i]]$cvm[resl[[i]]$lambda == resl[[i]]$lambda.1se]
      # $cvm is the cross validated measure in our case MSE
      # gets the smallest $cvm @ the lambda.1se location
    })
  } else {
    outl <- sapply(seq_along(resl), function(i){
      resl[[i]]$cvm[resl[[i]]$lambda == resl[[i]]$lambda.min]
    })
  }
  # ID of smallest $cvm measure ("mse") @lambda.1se accross all CV alphas-lambda model pairs
  l <- which.min(outl)          # which.min only operates on vectors => sapply()
  # Get alpha value with above ID
  alphas <- seq(alpha.min, alpha.max, length.out = length(resl)) # assuming equal length/by division within alpha range min-max 0-1 => 0.1 => 11
  opt.alpha <- alphas[[l]]

  # Extracts optimal lambda either .1se (suggested by GLMNET vignette more robust estimate see. Breiman 1984) or at lambda.min
  if(lambda.1se){opt.lambda <- resl[[l]]$lambda.1se}
  else{opt.lambda <- resl[[l]]$lambda.min}

  # Gets directly the ID (l) with min $cvm => needed for predict()
  opt.model <- resl[[which.min(outl)]] # same as resl[[l]]

  # Results
  res <- list(cvm.alpha.i.list = outl,
              opt.id = l,
              opt.alpha = opt.alpha,
              opt.lambda = opt.lambda,
              opt.mod = opt.model) # model object `glmnet.fit` is => ext.cvfit.v2$opt.mod$glmnet.fit
  return(res)
}


trainGLMNET <- function(y,
                        betas,
                        seed = 1234,
                        nfolds.cvglmnet = 10,
                        mc.cores = 2L,
                        parallel.cvglmnet = T,
                        alpha.min = 0, alpha.max = 1, by = 0.1){

  ## 1. Train RF for variable selection
  set.seed(seed,kind = "default")
  message("seed: ", seed)
  message("n: ", nrow(betas))  # n_patients
  message("cores: ", mc.cores)
  #getOption("mc.cores", mc.cores)#paste0(mc.cores, "L")) # getOption("mc.cores", detectCores()-2L)

  message("Start (concurrent) tuning of cv.glmnet hyperparameters alpha and lambda @  ", Sys.time())

  cvfit.glmnet.tuning <- subfunc_cvglmnet_tuner_mc_v2(x = betas,
                                                      y = y,
                                                      seed = seed,
                                                      nfolds = nfolds.cvglmnet,
                                                      family = "multinomial",
                                                      type.meas = "mse",
                                                      alpha.min = alpha.min, alpha.max = alpha.max, by = by,
                                                      n.lambda = 100,
                                                      lambda.min.ratio = 10^-6,
                                                      balanced.foldIDs = T,
                                                      mc.cores = mc.cores, # mc.cores is for the mclapply() shuffling through the alpha grid deafult 0-1 (x11)
                                                      parallel.comp = parallel.cvglmnet)


  # Extract permutation based importance measure # USE Version 2 of extractor!
  res.cvfit.glmnet.tuned <- subfunc_glmnet_mse_min_alpha_extractor(cvfit.glmnet.tuning, alpha.min = alpha.min, alpha.max = alpha.max)

  # Output hyperparameter results in the console
  message("Hyperparameter Tuning Results:",
          "\n Optimal alpha: ", res.cvfit.glmnet.tuned$opt.alpha,
          "\n Optimal lambda: ", res.cvfit.glmnet.tuned$opt.lambda)

  message("Re-fitting optimal/tuned model on data @ ", Sys.time())

  probs.glmnet.tuned <- predict(object = res.cvfit.glmnet.tuned$opt.mod,
                                newx = betas,
                                s = res.cvfit.glmnet.tuned$opt.mod$lambda.1se,
                                type = "response")[,,1]


  # Results
  res <- list(res.cvfit.glmnet.tuned$opt.mod,
              probs.glmnet.tuned,
              res.cvfit.glmnet.tuned)
  return(res)
}



