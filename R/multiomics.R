
#MOGOnet####

probeannotation <- function(platform = 450,
                            finalprobes){

  if(platform == 27){
    annotation <- 'ilmn12.hg19'
    array <- 'IlluminaHumanMethylation27k'
  }else if(platform == 450){
    annotation <- 'ilmn12.hg19'
    array <- 'IlluminaHumanMethylation450k'
  }else if(platform == 850){
    annotation <- 'ilm10b4.hg19'
    array <- 'IlluminaHumanMethylationEPIC'
  }else{
    cat('The parameter `platform` should be provided a value from 27, 450, and 850\n')
    return(NULL)
  }

  annopackage <- paste0(array, 'anno.', annotation)

  if(!(annopackage %in% installed.packages()[,'Package'])){
    cat(paste0('Package ', annopackage, ' is needed to run this function\n'))
    return(NULL)
  }

  if(!('AnnotationDbi' %in% installed.packages()[,'Package'])){
    cat('Package AnnotationDbi is needed to run this function\n')
    return(NULL)
  }

  if(!('org.Hs.eg.db' %in% installed.packages()[,'Package'])){
    cat(paste0('Package org.Hs.eg.db is needed to run this function\n'))
    return(NULL)
  }


  if(platform == 27){

    probeinfo <- IlluminaHumanMethylation27kanno.ilmn12.hg19::Other
    islandsinfo <- probeinfo[c("CPG_ISLAND_LOCATIONS", "CPG_ISLAND")]
    locinfo <- IlluminaHumanMethylation27kanno.ilmn12.hg19::Locations

    selectedcol <- c("Symbol", "Distance_to_TSS")
    genecol <- 'Symbol'
    featurecol <- 'Distance_to_TSS'
    islandcol <- 'CPG_ISLAND'
  }else if(platform == 450){

    probeinfo <- IlluminaHumanMethylation450kanno.ilmn12.hg19::Other
    islandsinfo <- IlluminaHumanMethylation450kanno.ilmn12.hg19::Islands.UCSC
    locinfo <- IlluminaHumanMethylation450kanno.ilmn12.hg19::Locations

    selectedcol <- c("UCSC_RefGene_Name", "UCSC_RefGene_Group")
    genecol <- 'UCSC_RefGene_Name'
    featurecol <- 'UCSC_RefGene_Group'
    islandcol <- 'Relation_to_Island'
  }else if(platform == 850){

    probeinfo <- IlluminaHumanMethylationEPICanno.ilm10b4.hg19::Other
    islandsinfo <- IlluminaHumanMethylationEPICanno.ilm10b4.hg19::Islands.UCSC
    locinfo <- IlluminaHumanMethylationEPICanno.ilm10b4.hg19::Locations

    selectedcol <- c("UCSC_RefGene_Name", "UCSC_RefGene_Group")
    genecol <- 'UCSC_RefGene_Name'
    featurecol <- 'UCSC_RefGene_Group'
    islandcol <- 'Relation_to_Island'
  }

  finalprobes <- finalprobes[finalprobes %in% row.names(probeinfo)]
  if(length(finalprobes) == 0){
    return(NULL)
  }


  probeinfo <- probeinfo[finalprobes,]
  probeinfodata <- as.data.frame(probeinfo)

  probeinfodata <- probeinfodata[selectedcol]

  islandsinfo <- islandsinfo[finalprobes,]
  islandsinfodata <- as.data.frame(islandsinfo)

  locinfo <- locinfo[finalprobes,]
  locinfo <- as.data.frame(locinfo)

  probeinfodata <- cbind(locinfo, probeinfodata, islandsinfodata)
  probeinfodata$Probe <- row.names(probeinfodata)
  row.names(probeinfodata) <- 1:nrow(probeinfodata)

  if(platform == 27){
    probeinfodata$ENTREZID <- probeinfo$Gene_ID
  }


  orgnizeprobeinfo <- function(colvec){
    parseelement <- function(element){
      elementlist <- unlist(strsplit(x = element, split = ';', fixed = TRUE))

      if(length(elementlist) == 0){
        elementlist <- ''
      }

      return(elementlist)
    }
    collist <- lapply(colvec, parseelement)
    return(collist)
  }

  genenamelist <- orgnizeprobeinfo(colvec = probeinfodata[,genecol])
  featurelist <- orgnizeprobeinfo(colvec = probeinfodata[,featurecol])
  poslist <- orgnizeprobeinfo(colvec = probeinfodata[,islandcol])

  genenamelistlens <- unlist(lapply(X = genenamelist, FUN = length))
  featurelistlens <- unlist(lapply(X = featurelist, FUN = length))
  poslistlens <- unlist(lapply(X = poslist, FUN = length))

  if(sum(genenamelistlens != 1) == 0 & platform == 27){
    probeinfodata$ENTREZID <- gsub(pattern = 'GeneID:', replacement = '',
                                   x = probeinfodata$ENTREZID, fixed = TRUE)
  }

  if(sum(genenamelistlens == 0) == 0){

    datnames <- colnames(probeinfodata)

    chrvec <- rep(probeinfodata[,1], times = genenamelistlens)
    locvec <- rep(probeinfodata[,2], times = genenamelistlens)
    strandvec <- rep(probeinfodata[,3], times = genenamelistlens)

    genenamevec <- unlist(genenamelist)
    featurevec <- unlist(featurelist)

    islandvec <- rep(probeinfodata[,6], times = genenamelistlens)
    posvec <- rep(probeinfodata[,7], times = genenamelistlens)
    probevec <- rep(probeinfodata[,8], times = genenamelistlens)

    if(platform == 27){

      geneidlist <- orgnizeprobeinfo(colvec = probeinfodata$ENTREZID)
      geneidvec <- unlist(geneidlist)
      geneidvec <- gsub(pattern = 'GeneID:', replacement = '',
                        x = geneidvec, fixed = TRUE)

      probeinfodata <- tryCatch({
        data.frame(chrvec, locvec, strandvec,
                   genenamevec,
                   featurevec,
                   islandvec, posvec, probevec,
                   geneidvec,
                   stringsAsFactors = FALSE)
      }, error = function(err){
        probeinfodata
      })

      names(probeinfodata) <- datnames[1:ncol(probeinfodata)]

      probeinfodata <- unique(probeinfodata)

    }else{

      unigeneidvec <- AnnotationDbi::select(x = org.Hs.eg.db::org.Hs.eg.db,
                                            keys = unique(genenamevec),
                                            columns = 'ENTREZID',
                                            keytype = 'SYMBOL')
      names(unigeneidvec)[1] <- selectedcol[1]

      probeinfodata <- tryCatch({
        data.frame(chrvec, locvec, strandvec,
                   genenamevec,
                   featurevec,
                   islandvec, posvec, probevec,
                   stringsAsFactors = FALSE)
      }, error = function(err){
        probeinfodata
      })

      names(probeinfodata) <- datnames[1:ncol(probeinfodata)]
      probeinfodata <- unique(probeinfodata)

      if(sum(!(unique(probeinfodata[,selectedcol[1]]) %in%
               unigeneidvec[,selectedcol[1]])) == 0){

        probeinfodata <- merge(probeinfodata, unigeneidvec,
                               by = selectedcol[1],
                               sort = FALSE)

      }

      probeinfodata <- unique(probeinfodata)

    }

    if('Distance_to_TSS' %in% datnames){

      probeinfodata$Distance_to_TSS <- as.integer(probeinfodata$Distance_to_TSS)

    }

    if('CPG_ISLAND' %in% datnames){

      probeinfodata$CPG_ISLAND <- as.logical(probeinfodata$CPG_ISLAND)

    }

  }

  if(platform == 27){

    resnames <- c('Probe', 'chr', 'pos', 'strand',
                  'CPG_ISLAND_LOCATIONS', 'CPG_ISLAND',
                  'Symbol', 'ENTREZID', 'Distance_to_TSS')

  }else{

    resnames <- c('Probe', 'chr', 'pos', 'strand',
                  'Islands_Name', 'Relation_to_Island',
                  'UCSC_RefGene_Name', 'ENTREZID', 'UCSC_RefGene_Group')

  }

  probeinfodata <- probeinfodata[,resnames[unique(c(1:7,
                                                    (ncol(probeinfodata) - 1), 9))]]

  row.names(probeinfodata) <- 1:nrow(probeinfodata)


  return(probeinfodata)

}


splitviews <- function(probes,
                       samplesize = 1000,
                       splitstandard = 'UCSC_RefGene_Group',
                       platform = 450,
                       seednum = 1234){

  probeanno <- probeannotation(platform = platform, finalprobes = probes)

  regiondat <- probeanno[c('Probe', splitstandard)]
  colnames(regiondat)[ncol(regiondat)] <- 'Group'

  grouplist <- list()

  if(splitstandard == 'Relation_to_Island'){

    islandset <- subset(regiondat, Group == 'Island')
    islandprobes <- unique(islandset$Probe)

    subsamplenum <- max(round(length(islandprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(islandprobes)/subsamplenum),
                         min(samplesize, length(islandprobes)))

    for(i in 1:subsamplenum){

      groupname <- paste0('islandprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = islandprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }




    nset <- subset(regiondat, (Group %in% c('N_Shelf', 'N_Shore')) &
                     !(Probe %in% islandprobes))
    nprobes <- unique(nset$Probe)

    subsamplenum <- max(round(length(nprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(nprobes)/subsamplenum),
                         min(samplesize, length(nprobes)))

    for(i in 1:subsamplenum){

      groupname <- paste0('nprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = nprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }


    sset <- subset(regiondat, (Group %in% c('S_Shelf', 'S_Shore')) &
                     !(Probe %in% c(islandprobes, nprobes)))
    sprobes <- unique(sset$Probe)

    subsamplenum <- max(round(length(sprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(sprobes)/subsamplenum),
                         min(samplesize, length(sprobes)))

    for(i in 1:subsamplenum){

      groupname <- paste0('sprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = sprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }


    openseaprobes <- setdiff(probes, c(islandprobes, nprobes, sprobes))

    subsamplenum <- max(round(length(openseaprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(openseaprobes)/subsamplenum),
                         min(samplesize, length(openseaprobes)))


    for(i in 1:subsamplenum){

      groupname <- paste0('openseaprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = openseaprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }


  }else{

    promoterset <- subset(regiondat, Group %in% c('TSS200', 'TSS1500', '1stExon'))
    promoterprobes <- unique(promoterset$Probe)

    subsamplenum <- max(round(length(promoterprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(promoterprobes)/subsamplenum),
                         min(samplesize, length(promoterprobes)))

    for(i in 1:subsamplenum){

      groupname <- paste0('promoterprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = promoterprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }

    bodyset <- subset(regiondat, (Group %in% c('Body', "3'UTR", "5'UTR", 'ExonBnd')) &
                        !(Probe %in% promoterprobes))
    bodyprobes <- unique(bodyset$Probe)

    subsamplenum <- max(round(length(bodyprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(bodyprobes)/subsamplenum),
                         min(samplesize, length(bodyprobes)))

    for(i in 1:subsamplenum){

      groupname <- paste0('bodyprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = bodyprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }

    otherprobes <- setdiff(probes, c(promoterprobes, bodyprobes))

    subsamplenum <- max(round(length(otherprobes)/samplesize), 1)

    subsamplesize <- max(ceiling(length(otherprobes)/subsamplenum),
                         min(samplesize, length(otherprobes)))

    for(i in 1:subsamplenum){

      groupname <- paste0('otherprobes', i)
      set.seed(seednum + i)
      grouplist[[groupname]] <- sample(x = otherprobes,
                                       size = subsamplesize,
                                       replace = FALSE)


    }

  }

  cat(paste0('Totally ', length(grouplist), ' views were generated, including\n'))

  for(i in 1:length(grouplist)){

    groupname <- names(grouplist)[i]

    groupsize <- length(grouplist[[i]])

    cat(paste0(groupname, ': ', groupsize, ' probes\n'))

  }

  return(grouplist)

}


samplenumadjust <- function(x,
                            samplenum,
                            seednum,
                            minsize,

                            viewstandard = NULL,
                            platform = 450){

  probes <- colnames(x)
  orisamplesize <- floor(length(probes)/samplenum)

  if(is.null(viewstandard)){

    addsize <- minsize - orisamplesize

    set.seed(seednum)
    probes <- sample(x = probes, size = length(probes), replace = FALSE)
    cutprobes <- probes[1:(orisamplesize*samplenum)]
    sampledprobelist <- split(cutprobes, seq(1, samplenum, 1))

    if(addsize > 0){

      for(i in 1:length(sampledprobelist)){

        baseprobes <- sampledprobelist[[i]]
        otherprobes <- setdiff(probes, baseprobes)

        if(length(otherprobes) < addsize){
          set.seed(seednum)
          addprobes <- sample(x = otherprobes, size = addsize, replace = TRUE)
        }else{
          set.seed(seednum)
          addprobes <- sample(x = otherprobes, size = addsize, replace = FALSE)
        }
        sampledprobelist[[i]] <- c(sampledprobelist[[i]], addprobes)
        set.seed(seednum)
        sampledprobelist[[i]] <- sample(x = sampledprobelist[[i]],
                                        size = length(sampledprobelist[[i]]),
                                        replace = FALSE)

      }

    }

  }else{

    sampledprobelist <- splitviews(probes = probes,
                                   samplesize = minsize,
                                   splitstandard = viewstandard,
                                   platform = platform,
                                   seednum = seednum)


  }

  return(sampledprobelist)

}



assignbaselearners <- function(x,
                               multiomicsnames,
                               samplenum = 10,
                               seednum = 1234,
                               minsize = 1000){

  datsizes <- as.vector(table(multiomicsnames)[unique(multiomicsnames)])
  datbasenums <- round(samplenum*(datsizes/sum(datsizes)))
  datbasenums[datbasenums == 0] <- 1

  sampledprobelist <- list()
  i <- 1
  for(i in 1:length(datbasenums)){

    sampledprobelist[[i]] <- samplenumadjust(x = x[, multiomicsnames == unique(multiomicsnames)[i], drop = FALSE],
                                             samplenum = datbasenums[i],
                                             seednum = seednum,
                                             minsize = minsize)
    names(sampledprobelist[[i]]) <- NULL
  }


  sampledprobelist <- do.call(c, sampledprobelist)

  return(sampledprobelist)

}


probesplit <- function(traindat,
                       testdat,

                       samplenum,
                       seednum,
                       minsize,
                       viewstandard = NULL,
                       platform = 450,

                       multiomicsnames = NULL){

  if(is.null(multiomicsnames)){

    grouplist <- samplenumadjust(x = traindat,
                                 samplenum = samplenum,
                                 seednum = seednum,
                                 minsize = minsize,

                                 viewstandard = viewstandard,
                                 platform = platform)


  }else{

    grouplist <- assignbaselearners(x = traindat,
                                    multiomicsnames = multiomicsnames,
                                    samplenum = samplenum,
                                    seednum = seednum,
                                    minsize = minsize)
  }


  if(is.null(names(grouplist))){
    names(grouplist) <- paste0('group', 1:length(grouplist))
  }

  traingroupdatlist <- list()
  testgroupdatlist <- list()
  i <- 1
  for(i in 1:length(grouplist)){

    groupname <- names(grouplist)[i]

    groupprobes <- grouplist[[i]]

    traingroupdat <- traindat[,groupprobes]
    testgroupdat <- testdat[,groupprobes]

    traingroupdatlist[[i]] <- traingroupdat
    testgroupdatlist[[i]] <- testgroupdat

    names(traingroupdatlist)[i] <- names(testgroupdatlist)[i] <- groupname

  }

  res <- list(traingroupdatlist = traingroupdatlist,
              testgroupdatlist = testgroupdatlist)

  return(res)

}

mogonet <- function(traingroupdatlist,
                    testgroupdatlist,
                    trainlabels,
                    testlabels = NULL,
                    pythonpath,
                    #mogonetpyfile = '/data/liuy47/nihcodes/mogonet_r.py',
                    mogonetpyfile,
                    K,
                    k,

                    num_epoch_pretrain,
                    num_epoch,
                    adj_parameter,
                    dim_he_list,

                    lr_e_pretrain = 1e-3,
                    lr_e = 5e-4,
                    lr_c = 1e-3,
                    seednum = 1234,
                    test_inverval = 50){

  if(!is.null(pythonpath)){

    Sys.setenv(RETICULATE_PYTHON = pythonpath)

  }

  #reticulate::use_python(pydir)

  reticulate::py_config()

  reticulate::source_python(mogonetpyfile)

  if(is.null(K)){

    K <- Sys.Date()

  }

  if(is.null(k)){

    k <- format(Sys.time(), '%T')
    k <- gsub(pattern = ':', replacement = '-', x = k)

  }

  K <- as.character(K)
  k <- as.character(k)

  num_class <- length(unique(c(trainlabels, testlabels)))
  testsamplenames <- row.names(testgroupdatlist[[1]])

  names(traingroupdatlist) <- names(testgroupdatlist) <- NULL

  if(is.null(testlabels)){

    testlabels <- numeric(nrow(testgroupdatlist[[1]]))


  }

  trainlabelmat <- matrix(as.numeric(trainlabels))
  testlabelmat <- matrix(as.numeric(testlabels))
  colnames(trainlabelmat) <- colnames(testlabelmat) <- 'labels'
  row.names(trainlabelmat) <- row.names(traingroupdatlist[[1]])
  row.names(testlabelmat) <- row.names(testgroupdatlist[[1]])
  trainlabelmat <- trainlabelmat - 1
  testlabelmat <- testlabelmat - 1

  traintestgroupdatlist <- list()

  for(i in 1:length(traingroupdatlist)){
    traintestgroupdatlist[[i]] <- rbind(traingroupdatlist[[i]],
                                        testgroupdatlist[[i]])

  }

  traintestlabelmat <- rbind(trainlabelmat, testlabelmat)

  #save(data_tr_list, file = 'data_tr_list_mogonet.RData')
  #save(data_te_list, file = 'data_te_list_mogonet.RData')
  #save(labels_tr, file = 'labels_tr_mogonet.RData')
  #save(labels_te, file = 'labels_te_mogonet.RData')

  mogonetres <- mogonet_r(data_tr_list = traingroupdatlist,
                          data_te_list = traintestgroupdatlist,
                          labels_tr = trainlabelmat,
                          labels_te = traintestlabelmat,
                          num_class = num_class,

                          num_epoch_pretrain = num_epoch_pretrain,
                          num_epoch = num_epoch,
                          lr_e_pretrain = lr_e_pretrain,
                          lr_e = lr_e,
                          lr_c = lr_c,
                          seednum = seednum,
                          test_inverval = test_inverval,
                          adj_parameter = adj_parameter,
                          dim_he_list = dim_he_list)

  mogonetres <- as.matrix(mogonetres)

  trainres <- mogonetres[1:nrow(trainlabelmat),]
  testres <- mogonetres[(nrow(trainlabelmat) + 1):nrow(mogonetres),]

  row.names(trainres) <- row.names(trainlabelmat)
  row.names(testres) <- row.names(testlabelmat)
  colnames(trainres) <- colnames(testres) <- levels(trainlabels)

  res <- list(trainres = trainres,
              testres = testres)

  return(res)


}

#eSVM####

## Cost selector subfunction

subfunc_eSVM_linear_C_selector <- function(results.cvfit.eSVM.linear.C.tuner,
                                           C.base = 10,
                                           C.min = -3,
                                           C.max = 3,
                                           n.CV = 5,
                                           verbose = TRUE){

  Costs.l <- as.list(C.base^(C.min:C.max))

  i <- 1

  # Print simplified version of accuracy for eventual manual selection
  res.cvfit.svm.accuracies <- sapply(seq_along(C.base^(C.min:C.max)), function(i){
    simplify2array(results.cvfit.eSVM.linear.C.tuner[[i]]$acc)})

  names(res.cvfit.svm.accuracies) <- paste0("Cost_", Costs.l)

  if(verbose){
    message("\nMatrix of all accuracies:")
    print(res.cvfit.svm.accuracies)
  }


  # Selection
  # Chooses the smallest C with highest accuracy among possible choices
  C.selected <- Costs.l[[which.max(res.cvfit.svm.accuracies)]]
  message("\nCost parameter with highest accuracy : C = ", C.selected, " ; ",
          "\n Note: If more than one maximal accuracy exists, C returns the smallest cost parameter with highest accuracy.",
          "\n Once C is large than a certain value, the obtained models have similar performances")
  res <- list(C.selected = C.selected,
              accuracies = res.cvfit.svm.accuracies)

  return(res)

}


svmensemble <- function(svmlist){

  baselearnerlist <- list()
  errlist <- c()
  i <- 0
  while(length(svmlist) > 0){
    i <- i + 1
    baselearner <- svmlist[[1]]
    svmlist[[1]] <- NULL
    totacc <- baselearner$tot.accuracy
    err <- 1 - totacc/100

    if(err < 0.5){

      baselearnerlist[[paste0('SVM', i)]] <- baselearner
      errlist <- c(errlist, err)

    }else{
      next()
    }

  }

  if(length(errlist) < 1){

    return(NULL)

  }

  names(baselearnerlist) <- paste0('SVM', 1:length(baselearnerlist))

  res <- list(baselearners = baselearnerlist,
              errs = errlist)

  return(res)

}


eSVMpredict <- function(eSVMmod,
                        x,
                        cores){

  singlepres <- function(i,
                         modlist = eSVMmod,
                         newdat = x){

    baselearner <- modlist$baselearners[[i]]
    normweight <- modlist$normweights[i]
    baselearnerfeatures <- colnames(baselearner$SV)
    dat <- newdat[,baselearnerfeatures, drop = FALSE]

    baselearnerres <- tryCatch({

      predict(object = baselearner,
              newdata = dat,
              decision.values = TRUE,
              # decision values of all binary classif. in multiclass setting are returned.
              probability = TRUE)

    }, error = function(err){

      e1071:::predict.svm(object = baselearner,
                          newdata =  dat,
                          decision.values = TRUE,
                          # decision values of all binary classif. in multiclass setting are returned.
                          probability = TRUE)

    })

    baselearnerscores <- attr(baselearnerres, 'probabilities')
    baselearnerscores <- normweight*baselearnerscores

    return(baselearnerscores)

  }

  if(cores == 1){

    iseqs <- 1:length(eSVMmod$baselearners)

    baselearnerreslist <- list()

    for(i in iseqs){

      baselearnerreslist[[i]] <- singlepres(i = i,
                                            modlist = eSVMmod,
                                            newdat = x)

    }

  }else{

    iseqs <- 1:length(eSVMmod$baselearners)

    #library(doParallel)

    threads <- parallel::detectCores()
    cl <- parallel::makeCluster(min(cores, threads))

    doParallel::registerDoParallel(cl)

    #date()
    `%dopar%` <- foreach::`%dopar%`
    baselearnerreslist <- foreach::foreach(i = iseqs,
                                           #.export = ls(name = globalenv())) %dopar% {
                                           .export = NULL) %dopar% {
                                             singlepres(i,
                                                        modlist = eSVMmod,
                                                        newdat = x)
                                           }

    parallel::stopCluster(cl)

    unregister_dopar()

  }

  scores <- Reduce(`+`, baselearnerreslist)

  return(scores)

}


eSVM <- function(x,
                 y,
                 type,
                 kernel = 'linear',
                 cost = 1,
                 cross,
                 probability = TRUE,
                 fitted = TRUE,
                 seednum = 1234,
                 samplenum = 10,
                 cores = 1,

                 minsize = 1000,
                 viewstandard = NULL,
                 platform = 450,


                 multiomicsnames = NULL,
                 featurescale = TRUE,
                 weighted = FALSE){

  if(is.null(multiomicsnames)){

    sampledprobelist <- samplenumadjust(x = x,
                                        samplenum = samplenum,
                                        seednum = seednum,
                                        minsize = minsize,

                                        viewstandard = viewstandard,
                                        platform = platform)

  }else{

    sampledprobelist <- assignbaselearners(x = x,
                                           multiomicsnames = multiomicsnames,
                                           samplenum = samplenum,
                                           seednum = seednum,
                                           minsize = minsize)

  }

  singlesvm <- function(j,
                        x,
                        y,
                        sampledprobelist,
                        featurescale,
                        type,
                        kernel,
                        cost,
                        cross,
                        probability,
                        fitted,
                        weighted = weighted){

    simpleboot <- function(resrange,
                           sampleseed,
                           samplevar){

      resrangetmp <- data.frame(sampleid = row.names(samplevar), Response = resrange)
      sampleseedtmp <- sampleseed
      samplevartmp <- samplevar

      groups <- resrangetmp[c('sampleid', 'Response')]
      row.names(groups) <- 1:nrow(groups)
      samplesize <- nrow(groups)

      grouplist <- split(x = groups, f = groups$Response)
      bootressampleididx <- c()
      i <- 1
      for(i in 1:length(grouplist)){

        singlegroup <- grouplist[[i]]
        singlegroupsize <- nrow(singlegroup)

        set.seed(sampleseed)
        singlegroupsampleididx <- sample(x = row.names(singlegroup),
                                         size = singlegroupsize, replace = TRUE)
        singlegroupsampleididx <- as.numeric(singlegroupsampleididx)
        bootressampleididx <- c(bootressampleididx, singlegroupsampleididx)

      }

      set.seed(sampleseed)
      bootressampleididx <- sample(x = bootressampleididx,
                                   size = samplesize,
                                   replace = FALSE)
      bootressampleid <- groups$sampleid[bootressampleididx]
      rm(grouplist)

      bootsamplevar <- samplevar[bootressampleid,]
      bootgroups <- groups[bootressampleididx,]

      suffix <- rep('', nrow(bootgroups))
      suffix[grepl(pattern = '\\.', x = row.names(bootgroups))] <-
        substring(row.names(bootgroups),
                  regexpr('\\.', row.names(bootgroups)))[grepl(pattern = '\\.',
                                                               x = row.names(bootgroups))]

      bootgroups$sampleid <- paste0(bootgroups$sampleid, suffix)
      row.names(bootgroups) <- 1:nrow(bootgroups)
      row.names(bootsamplevar) <- bootgroups$sampleid

      bootgroups <- bootgroups[order(bootgroups$Response),]
      bootsamplevar <- bootsamplevar[bootgroups$sampleid,]
      row.names(bootgroups) <- 1:nrow(bootgroups)


      names(bootgroups)[1] <- c('resampleid')
      final <- list(resampleresponse = bootgroups,
                    resamplevars = bootsamplevar)


      return(final)

    }

    sampledprobes <- sampledprobelist[[j]]

    bootdat <- simpleboot(resrange = y,
                          sampleseed = j,
                          samplevar = x)

    if(weighted == TRUE){
      classweights <- 100/table(bootdat$resampleresponse$Response)
    }else{
      classweights <- NULL
    }

    svmres <- e1071::svm(x = bootdat$resamplevars[,sampledprobes],
                         y = bootdat$resampleresponse$Response,
                         scale = featurescale,
                         type = type,
                         kernel = kernel,
                         cost = cost,
                         cross = cross,
                         probability = probability,
                         fitted = fitted,

                         class.weights = classweights)
    colnames(svmres$SV) <- sampledprobes

    return(svmres)

  }

  samplenum <- length(sampledprobelist)

  if(cores == 1){

    svmreslist <- list()

    for(j in 1:samplenum){

      singlesvmres <- singlesvm(j = j,
                                x = x,
                                y = y,
                                sampledprobelist = sampledprobelist,
                                featurescale = featurescale,
                                type = type,
                                kernel = kernel,
                                cost = cost,
                                cross = cross,
                                probability = probability,
                                fitted = fitted,
                                weighted = weighted)

      svmreslist[[j]] <- singlesvmres

      cat(paste0('j = ', j, '\n'))

    }

  }else{


    jseqs <- 1:samplenum

    #library(doParallel)

    threads <- parallel::detectCores()
    cl <- parallel::makeCluster(min(cores, threads))

    doParallel::registerDoParallel(cl)

    #date()
    `%dopar%` <- foreach::`%dopar%`
    svmreslist <- foreach::foreach(j = jseqs,
                                   #.export = ls(name = globalenv())) %dopar% {
                                   .export = NULL) %dopar% {
                                     singlesvm(j,
                                               x = x, y = y,
                                               sampledprobelist = sampledprobelist,
                                               featurescale = featurescale, type = type,
                                               kernel = kernel, cost = cost,
                                               cross = cross,
                                               probability = probability,
                                               fitted = fitted,
                                               weighted = weighted)
                                   }

    parallel::stopCluster(cl)

    unregister_dopar()

  }

  svmensembleres <- svmensemble(svmlist = svmreslist)

  if(is.null(svmensembleres)){

    stop('No SVM base learner can be fitted with an error < 0.5')

  }

  svmensembleerrs <- svmensembleres$errs

  if(sum(svmensembleerrs == 0) >= 1){

    if(sum(svmensembleerrs != 0) >= 1){
      othererrs <- svmensembleerrs[svmensembleerrs != 0]
      svmensembleerrs[svmensembleerrs == 0] <- min(othererrs)
    }else{
      svmensembleerrs <- rep(0.0001, length(svmensembleerrs))
    }

  }

  weights <- 0.5*log((1 - svmensembleerrs)/svmensembleerrs)
  normweights <- weights/sum(weights)

  svmensembleres$weights <- weights
  svmensembleres$normweights <- normweights

  scores <- eSVMpredict(eSVMmod = svmensembleres,
                        x = x,
                        cores = cores)

  pres <- colnames(scores)[apply(scores, 1, which.max)]
  acc <- sum(pres == y)/length(y)


  res <- list()
  res$model <- svmensembleres
  res$scores <- scores
  res$pres <- pres
  res$acc <- acc

  return(res)

}


subfunc_eSVM_train_tuner_mc <- function(data.xTrain,
                                        target.yTrain,
                                        mod.type = "C-classification",
                                        kernel. = "linear",
                                        C.base = 10,
                                        C.min = -3,
                                        C.max = 3,
                                        n.CV = 5,
                                        verbose = T,
                                        seed = 1234,
                                        #parallel = T,
                                        mc.cores = 4L,
                                        samplenum = 10,
                                        minsize = 1000,
                                        viewstandard = NULL,
                                        platform = 450,

                                        multiomicsnames = NULL,
                                        featurescale = TRUE,
                                        weighted = FALSE){

  # Cost C grid + give feedback and Sys.time
  Cost.l <- as.list(C.base^(C.min:C.max))
  message("\nCost (C) = ", paste(simplify2array(Cost.l), sep = " ", collapse = " ; "),
          " ; \nNr. of iterations: ", length(Cost.l),
          "\nStart at ", Sys.time())
  # Predefine empty list for results
  cvfit.eSVM.linear.C.tuner <- list()

  for(i in seq_along(Cost.l)){

    cvfit.eSVM.linear.C.tuner[[i]] <- eSVM(x = data.xTrain,
                                           y = target.yTrain,
                                           type = mod.type,
                                           kernel = kernel.,
                                           cost = Cost.l[[i]],
                                           cross = n.CV,
                                           probability = TRUE,
                                           fitted = TRUE,
                                           seednum = seed + 1,
                                           samplenum = samplenum,
                                           cores = mc.cores,
                                           minsize = minsize,
                                           viewstandard = viewstandard,
                                           platform = platform,

                                           multiomicsnames = multiomicsnames,
                                           featurescale = featurescale,
                                           weighted = weighted)

  }

  print(Sys.time())
  return(cvfit.eSVM.linear.C.tuner)

}

### Training & tuning function

train_eSVM <- function(y,
                       betas.Train,
                       seed,
                       mc.cores,
                       nfolds = 5,
                       C.base = 10,
                       C.min = -3,
                       C.max = 3,
                       samplenum = 10,
                       mod.type = "C-classification",
                       minsize = 1000,
                       viewstandard = NULL,
                       platform = 450,

                       multiomicsnames = NULL,
                       featurescale = TRUE,
                       weighted = FALSE){

  ## 1. Crossvalidate SVM/LiblineaR - Cost parameter for optimal
  set.seed(seed, kind = "default")
  message("seed: ", seed)
  message("n: ", nrow(betas.Train))  # n_patients

  message("\nTuning eSVM linear kernel: hyperparameter C (cost) ... ", Sys.time())


  cvfit.eSVM.linear.C.tuner <- subfunc_eSVM_train_tuner_mc(data.xTrain = betas.Train,
                                                           target.yTrain = y,
                                                           mod.type = mod.type,
                                                           kernel. = "linear",
                                                           C.base = C.base,
                                                           C.min = C.min,
                                                           C.max = C.max,
                                                           n.CV = nfolds,
                                                           verbose = T,
                                                           seed = seed,
                                                           #parallel = T,
                                                           mc.cores = mc.cores,
                                                           samplenum = samplenum,
                                                           minsize = minsize,
                                                           viewstandard = viewstandard,
                                                           platform = platform,

                                                           multiomicsnames = multiomicsnames,
                                                           featurescale = featurescale,
                                                           weighted = weighted)





  # Extract optimal C or smallest C with highest accuracy
  C.tuned.cv <-  subfunc_eSVM_linear_C_selector(results.cvfit.eSVM.linear.C.tuner = cvfit.eSVM.linear.C.tuner,
                                                C.base = C.base,
                                                C.min = C.min,
                                                C.max = C.max,
                                                n.CV = nfolds,
                                                verbose = TRUE)


  # Provide message with value
  message(paste0("Optimal cost (C) parameter: ", C.tuned.cv$C.selected))

  idx <- grep(pattern = paste0('_', C.tuned.cv$C.selected), x = names(C.tuned.cv$accuracies))

  modfit.eSVM.linear.train <- cvfit.eSVM.linear.C.tuner[[idx]]

  pred.scores.trainfit.eSVM <- modfit.eSVM.linear.train$scores

  # Results
  res <- list(modfit.eSVM.linear.train$model,
              pred.scores.trainfit.eSVM,
              C.tuned.cv)
  return(res)

}

#eNeural#####

neuralensemble <- function(neurallist){

  baselearnerlist <- list()
  errlist <- c()
  modelidlist <- c()
  i <- 0
  while(length(neurallist) > 0){
    i <- i + 1
    baselearner <- neurallist[[1]]
    neurallist[[1]] <- NULL

    modelid <- baselearner@model_id

    #Get model error
    err <- as.numeric(baselearner@model$cross_validation_metrics_summary['err', 'mean'])

    if(err < 0.5){

      baselearnerlist[[paste0('Neural', i)]] <- baselearner
      errlist <- c(errlist, err)
      modelidlist <- c(modelidlist, modelid)

    }else{
      next()
    }

  }

  if(length(errlist) < 1){

    return(NULL)

  }

  names(baselearnerlist) <- paste0('Neural', 1:length(baselearnerlist))

  res <- list(baselearners = baselearnerlist,
              errs = errlist,
              modelids = modelidlist)

  return(res)

}

eNeuralpredict <- function(eNeuralmod,
                           x,
                           cores,
                           baselearnerpath = NULL){

  singlepres <- function(i,
                         modlist,
                         newdat,
                         baselearnerpath = NULL){


    baselearner <- modlist$baselearners[[i]]
    normweight <- modlist$normweights[i]

    baselearnerfeatures <- baselearner@parameters$x
    dat <- newdat[,baselearnerfeatures, drop = FALSE]

    if(!is.null(baselearnerpath)){

      basepathes <- dir(baselearnerpath)
      basepath <- basepathes[i]
      basepath <- file.path(baselearnerpath, basepath)

      h2o::h2o.loadModel(path = basepath)

    }


    baselearnerscores <- tryCatch({

      predict(baselearner,
              h2o::as.h2o(dat),
              type = "prob")

    }, error = function(err){

      NULL

    })

    if(is.null(baselearnerscores)){

      datt <- dat
      datt <- as.data.frame(datt, stringsAsFactors = FALSE)

      labels <- row.names(h2o::h2o.confusionMatrix(modlist$baselearners[[i]]))
      labels <- labels[-length(labels)]

      set.seed(i)
      datt$Response <- sample(x = h2o::as.factor(labels),
                              size = nrow(datt), replace = TRUE)

      datt <- datt[c('Response', colnames(datt)[-ncol(datt)])]

      baselearnerscores <- predict(baselearner,
                                   h2o::as.h2o(datt),
                                   type = "prob")

    }

    baselearnerscores <- as.data.frame(baselearnerscores)
    baselearnerscores <- baselearnerscores[-1]
    baselearnerscores <- as.matrix(baselearnerscores)

    #Change to the original labels and sample names
    confusionmat <- baselearner@model$training_metrics@metrics$cm$table
    labels <- row.names(confusionmat)[-nrow(confusionmat)]
    colnames(baselearnerscores) <- labels
    row.names(baselearnerscores) <- row.names(dat)


    baselearnerscores <- normweight*baselearnerscores

    return(baselearnerscores)

  }

  h2o::h2o.init(nthreads = cores)


  iseqs <- 1:length(eNeuralmod$baselearners)

  baselearnerreslist <- list()

  for(i in iseqs){

    baselearnerreslist[[i]] <- singlepres(i = i,
                                          modlist = eNeuralmod,
                                          newdat = x,
                                          baselearnerpath = baselearnerpath)

  }


  scores <- Reduce(`+`, baselearnerreslist)

  #h2o::h2o.shutdown(prompt = FALSE)
  #h2o::h2o.init(nthreads = cores)

  return(scores)

}

eNeural <- function(x,
                    y,
                    cross,
                    seednum = 1234,
                    samplenum = 10,
                    cores = 1,

                    predefinedhidden = NULL,
                    maxepochs = 10,

                    activation = 'Rectifier',
                    momentum_start = 0,
                    rho = 0.99,

                    gridsearch = FALSE,
                    savefile = FALSE,

                    minsize = 1000,
                    viewstandard = NULL,
                    platform = 450,

                    multiomicsnames = NULL){

  if(is.null(multiomicsnames)){

    sampledprobelist <- samplenumadjust(x = x,
                                        samplenum = samplenum,
                                        seednum = seednum,
                                        minsize = minsize,

                                        viewstandard = viewstandard,
                                        platform = platform)

  }else{

    sampledprobelist <- assignbaselearners(x = x,
                                           multiomicsnames = multiomicsnames,
                                           samplenum = samplenum,
                                           seednum = seednum,
                                           minsize = minsize)

  }




  gethiddenlayer <- function(orineuronnum,
                             predefinednum = NULL){

    if(orineuronnum >= 10000){

      n <- floor(orineuronnum/100)

      hiddennum1 <- n*10

      if(!is.null(predefinednum)){

        hiddennum1 <- predefinednum

      }

      hiddennum2 <- round(hiddennum1/4)

    }else if(orineuronnum > 100){

      n <- floor(orineuronnum/100)

      hiddennum1 <- n*10

      if(!is.null(predefinednum)){

        hiddennum1 <- predefinednum

      }

      hiddennum2 <- round(hiddennum1/2)


    }else if(orineuronnum > 10){

      n <- floor(orineuronnum/10)

      hiddennum1 <- n*1

      if(!is.null(predefinednum)){

        hiddennum1 <- predefinednum

      }

      hiddennum2 <- round(hiddennum1/2)

    }else{

      hiddennum1 <- hiddennum2 <- orineuronnum

      if(!is.null(predefinednum)){
        hiddennum1 <- hiddennum2 <- predefinednum
      }

    }

    res <- list(largenum = hiddennum1,
                smallnum = hiddennum2)


  }


  basicNeural <- function(oritrain,
                          oripdtrain,
                          cvfolds,

                          cores,
                          randseed,
                          predefinedhidden = NULL,
                          maxepochs = 10,

                          activation = 'Rectifier',
                          momentum_start = 0,
                          rho = 0.99,

                          gridsearch = TRUE,

                          savefile = FALSE){

    #h2o::h2o.init(nthreads = cores)
    #Attempts to start and/or connect to an h2o instance
    #nthreads = -1 means use all CPUs on the host (Default)

    #h2o::h2o.removeAll() #Remove all objects on the h2o cluster

    oridat <- oritrain
    oridat <- as.data.frame(oridat, stringsAsFactors = FALSE)
    oridat$Response <- as.character(oripdtrain)
    oridat <- oridat[c('Response', colnames(oridat)[-ncol(oridat)])]

    h2odat <- h2o::as.h2o(oridat)




    h2odat[,1] <- h2o::as.factor(h2odat[,1])




    h2otrain <- h2odat[1:nrow(oritrain),]

    if(is.null(predefinedhidden)){
      hiddensets <- gethiddenlayer(orineuronnum = ncol(h2odat) - 1,
                                   predefinednum = NULL)
      if(gridsearch == TRUE){
        hidden <- list(c(hiddensets$largenum, hiddensets$smallnum, hiddensets$largenum),
                       c(hiddensets$largenum, hiddensets$largenum),
                       c(hiddensets$largenum))
      }else{
        hidden <- c(hiddensets$largenum, hiddensets$largenum)
      }

    }else{
      hidden <- predefinedhidden
    }


    K <- Sys.Date()
    K <- gsub(pattern = '-', replacement = '_', x = K)
    k <- format(Sys.time(), '%T')
    k <- gsub(pattern = ':', replacement = '_', x = k)

    K <- as.character(K)
    k <- as.character(k)


    if(gridsearch == TRUE){

      gridname <- paste0('deeplearning_', K, '_', k, '_grid')

      epochs <- round(10^seq(1, log10(maxepochs)))
      epochs <- epochs[epochs < maxepochs]
      epochs <- unique(c(epochs, maxepochs))
      epochs <- epochs[!is.na(epochs)]
      epochs <- unique(epochs)

      search_criteria <- list(strategy = 'RandomDiscrete',
                              max_models = 100,
                              stopping_rounds = 5,
                              stopping_tolerance = 1e-2,
                              seed = randseed)

      activation <- activation

      hyper_params <- list(

        activation = activation,
        epochs = epochs,

        hidden = hidden,

        train_samples_per_iteration = c(0,-2),

        momentum_start = momentum_start,
        rho = rho

      )

      deeplearning_grid <- h2o::h2o.grid(

        algorithm = 'deeplearning',
        model_id = gridname,

        x = 2:ncol(h2odat),
        y = 1,
        training_frame = h2otrain,
        #validation_frame = h2otest,
        seed = randseed,

        nfolds = cvfolds,
        hyper_params = hyper_params,

        score_duty_cycle = 0.025,
        #Maximum duty cycle fraction for scoring
        #(lower: more training, higher: more scoring).
        #Defaults to 0.1.
        #Don't score more than 2.5% of the wall time
        #max_w2=10,
        #Can help improve stability for Rectifier

        variable_importances = TRUE,
        export_weights_and_biases = TRUE,
        standardize = TRUE,
        stopping_metric = 'misclassification',
        stopping_tolerance = 1e-2,
        #stop when logloss does not improve by >=1% for 2 scoring events
        stopping_rounds = 2,

        search_criteria = search_criteria,

        parallelism = cores

      )

      grid <- h2o::h2o.getGrid(gridname,
                               sort_by = 'logloss',
                               decreasing = FALSE)

      #grid@summary_table[1,]

      deeplearning_model <- h2o::h2o.getModel(grid@model_ids[[1]])
      #model with lowest MSE or logloss

    }else{

      modelname <- paste0('deeplearning_', K, '_', k, '_model')

      deeplearning_model <- h2o::h2o.deeplearning(

        model_id = modelname,

        x = 2:ncol(h2odat),
        y = 1,
        training_frame = h2otrain,
        #validation_frame = h2otest,
        seed = randseed,

        nfolds = cvfolds,

        hidden = hidden,
        activation = activation,
        epochs = maxepochs,
        train_samples_per_iteration = -2,

        momentum_start = momentum_start,
        rho = rho,

        score_duty_cycle = 0.025,
        #Maximum duty cycle fraction for scoring
        #(lower: more training, higher: more scoring).
        #Defaults to 0.1.
        #Don't score more than 2.5% of the wall time
        #max_w2=10,
        #Can help improve stability for Rectifier

        variable_importances = TRUE,
        export_weights_and_biases = TRUE,
        standardize = TRUE,
        stopping_metric = 'misclassification',
        stopping_tolerance = 1e-2,
        #stop when logloss does not improve by >=1% for 2 scoring events
        stopping_rounds = 2

      )

    }

    if(savefile == TRUE){

      tag <- Sys.time()
      tag <- gsub(pattern = ' ', replacement = '_', x = tag)
      tag <- gsub(pattern = ' .*$', replacement = '', x = tag)
      tag <- gsub(pattern = ':', replacement = '_', x = tag)

      h2o::h2o.saveModel(object = deeplearning_model,
                         path = paste0('./deeplearning_mod_', tag))


    }

    #h2o::h2o.shutdown(prompt = FALSE)
    #h2o::h2o.init(nthreads = cores)

    return(deeplearning_model)

  }

  singleNeural <- function(j,
                           x,
                           y,
                           cross,
                           seednum = 1234,
                           cores = 1,

                           sampledprobelist,

                           predefinedhidden = NULL,
                           maxepochs = 10,

                           activation = 'Rectifier',
                           momentum_start = 0,
                           rho = 0.99,

                           gridsearch = TRUE,
                           savefile = FALSE){

    simpleboot <- function(resrange,
                           sampleseed,
                           samplevar){

      resrangetmp <- data.frame(sampleid = row.names(samplevar), Response = resrange)
      sampleseedtmp <- sampleseed
      samplevartmp <- samplevar

      groups <- resrangetmp[c('sampleid', 'Response')]
      row.names(groups) <- 1:nrow(groups)
      samplesize <- nrow(groups)

      grouplist <- split(x = groups, f = groups$Response)
      bootressampleididx <- c()
      i <- 1
      for(i in 1:length(grouplist)){

        singlegroup <- grouplist[[i]]
        singlegroupsize <- nrow(singlegroup)

        set.seed(sampleseed)
        singlegroupsampleididx <- sample(x = row.names(singlegroup),
                                         size = singlegroupsize, replace = TRUE)
        singlegroupsampleididx <- as.numeric(singlegroupsampleididx)
        bootressampleididx <- c(bootressampleididx, singlegroupsampleididx)

      }

      set.seed(sampleseed)
      bootressampleididx <- sample(x = bootressampleididx, size = samplesize, replace = FALSE)
      bootressampleid <- groups$sampleid[bootressampleididx]
      rm(grouplist)

      bootsamplevar <- samplevar[bootressampleid,]
      bootgroups <- groups[bootressampleididx,]

      suffix <- rep('', nrow(bootgroups))
      suffix[grepl(pattern = '\\.', x = row.names(bootgroups))] <-
        substring(row.names(bootgroups),
                  regexpr('\\.', row.names(bootgroups)))[grepl(pattern = '\\.',
                                                               x = row.names(bootgroups))]

      bootgroups$sampleid <- paste0(bootgroups$sampleid, suffix)
      row.names(bootgroups) <- 1:nrow(bootgroups)
      row.names(bootsamplevar) <- bootgroups$sampleid

      bootgroups <- bootgroups[order(bootgroups$Response),]
      bootsamplevar <- bootsamplevar[bootgroups$sampleid,]
      row.names(bootgroups) <- 1:nrow(bootgroups)


      names(bootgroups)[1] <- c('resampleid')
      final <- list(resampleresponse = bootgroups,
                    resamplevars = bootsamplevar)


      return(final)

    }

    sampledprobes <- sampledprobelist[[j]]

    bootdat <- simpleboot(resrange = y,
                          sampleseed = j,
                          samplevar = x)

    neuralres <- basicNeural(oritrain = bootdat$resamplevars[,sampledprobes],
                             oripdtrain = bootdat$resampleresponse$Response,
                             cvfolds = cross,
                             randseed = seednum,
                             cores = cores,

                             predefinedhidden = predefinedhidden,
                             maxepochs = maxepochs,

                             activation = activation,
                             momentum_start = momentum_start,
                             rho = rho,

                             gridsearch = gridsearch,
                             savefile = savefile)

    return(neuralres)

  }

  h2o::h2o.init(nthreads = cores)

  samplenum <- length(sampledprobelist)

  neuralreslist <- list()

  for(j in 1:samplenum){

    singleneuralres <- singleNeural(j = j,
                                    x = x,
                                    y = y,
                                    cross = cross,
                                    seednum = seednum,
                                    cores = cores,

                                    sampledprobelist = sampledprobelist,

                                    predefinedhidden = predefinedhidden,
                                    maxepochs = maxepochs,

                                    activation = activation,
                                    momentum_start = momentum_start,
                                    rho = rho,

                                    gridsearch = gridsearch,
                                    savefile = savefile)

    neuralreslist[[j]] <- singleneuralres

    cat(paste0('j = ', j))

  }



  neuralensembleres <- neuralensemble(neurallist = neuralreslist)

  if(is.null(neuralensembleres)){

    stop('No Neural base learner can be fitted with an error < 0.5')

  }

  neuralensembleerrs <- neuralensembleres$errs

  if(sum(neuralensembleerrs == 0) >= 1){

    if(sum(neuralensembleerrs != 0) >= 1){
      othererrs <- neuralensembleerrs[neuralensembleerrs != 0]
      neuralensembleerrs[neuralensembleerrs == 0] <- min(othererrs)
    }else{
      neuralensembleerrs <- rep(0.0001, length(neuralensembleerrs))
    }

  }

  weights <- 0.5*log((1 - neuralensembleres$errs)/neuralensembleres$errs)
  normweights <- weights/sum(weights)

  neuralensembleres$weights <- weights
  neuralensembleres$normweights <- normweights

  scores <- eNeuralpredict(eNeuralmod = neuralensembleres,
                           x = x,
                           cores = cores)

  pres <- colnames(scores)[apply(scores, 1, which.max)]
  acc <- sum(pres == y)/length(y)


  res <- list()
  res$model <- neuralensembleres
  res$scores <- scores
  res$pres <- pres
  res$acc <- acc

  #h2o::h2o.shutdown(prompt = FALSE)
  #h2o::h2o.init(nthreads = cores)

  return(res)

}





#Featureselection####

#Get limma diff sites list with each element as a comparison result between
#a specific disease sample and other non samples
#simpds data.frame must contain a column named "label"
diffsites <- function(dat,
                      simpds,
                      padjcut = 0.05,
                      xcutoff = 0.1,
                      cutnum = 10000){

  vars <- colnames(simpds)
  vars <- paste(vars, collapse = ' + ')

  #library(limma)

  diseases <- table(simpds$label)
  diseases <- diseases[diseases != 0]
  diseases <- diseases[order(-diseases)]
  diseases <- names(diseases)
  samplediseases <- simpds$label

  i <- 1
  sigg.limmas <- list()

  for(i in 1:length(diseases)){

    disease <- diseases[i]
    diseasecount <- sum(samplediseases == disease)


    simpds$label[samplediseases != disease] <- 'Control'
    simpds$label[samplediseases == disease] <- 'Disease'

    design <- model.matrix(as.formula(paste0('~ ', vars)),
                           data = simpds)

    #betasub <- dat[,simpds$Sample_Name]

    fit1 <- limma::lmFit(dat, design)
    fit1 <- limma::eBayes(fit1)

    allg.limma <- limma::topTable(fit1, coef=2, n=dim(fit1)[1])




    sigg.limma <- subset(allg.limma, (adj.P.Val < padjcut) &
                           (logFC > xcutoff | logFC < -xcutoff))

    sigg.limma$probes <- row.names(sigg.limma)
    sigg.limmas[[i]] <- sigg.limma


    cat(paste0('For group ', disease, ', significant limma probes = ', nrow(sigg.limma), '\n'))

  }

  sigg.limmas <- do.call(rbind, sigg.limmas)

  if(nrow(sigg.limmas) == 0){
    volcanofeatures <- NULL
  }else{
    sigg.limmas <- sigg.limmas[order(sigg.limmas$adj.P.Val, -abs(sigg.limmas$logFC)), , drop = FALSE]
    volcanofeatures <- unique(sigg.limmas$probes)
    volcanofeatures <- volcanofeatures[1:min(length(volcanofeatures), cutnum)]
  }

  return(volcanofeatures)

}


#'Select features for classification model construction
#'
#'Select features for classification model construction via selecting the top
#'variable features, using \code{SCMER}, or \code{limma}.
#'
#'@param y.. The true labels of the samples. Can be a vector, factor, or NULL.
#'  If it is a vector or factor, each element is a label for a sample and the
#'  element order in it should be the same as the sample order in the sample
#'  data provided by the parameter \code{betas..}. This is not necessary for
#'  the top variable and \code{SCMER} feature selection, and in these cases,
#'  it can be set as NULL.
#'@param betas.. The beta value matrix of the samples. Each row is one sample
#'  and each column is one feature. It can also be set as NULL, and in this
#'  case, the function will load the data via the directory \code{betas.path}.
#'  The absolute path of the file of these data should be provided by the
#'  parameter \code{betas.path}.
#'@param betas.path If the parameter \code{betas..} is NULL, this parameter is
#'  necessary to provide the file path of the betas matrix data, so that the
#'  function will load the data from this path. It should be an absolute path
#'  string, and the file should be an .rds file.
#'@param subset.CpGs The feature selection method. It can be a numeric number
#'  such as 10000, and then the top 10000 most variable features of the data
#'  will be selected. It can also be the string "limma", so that \code{limma}
#'  will be used to select the significantly differential features between a
#'  sample class and all other samples. The differential ones should fulfill
#'  the condition that their adjusted p-value < \code{padjcut} (default value
#'  is 0.05), and the betas value difference between the sample class and all
#'  other samples should be > \code{xcutoff}, or < -\code{xcutoff} (default is
#'  0.1). After the differential features for each class have been selected by
#'  \code{limma}, they will be mixed together and ordered according to their
#'  adjusted p-value and the absolute of beta value difference, and then the
#'  top \code{cutnum} ones (default number is 10000) will be selected and as
#'  the final features. Additionally, if there are any confounding factors in
#'  the dataset need to be removed, they should be provided via the parameter
#'  \code{confoundings} and \code{limma} will select the features after these
#'  confoundings have been adjusted. The parameter \code{confoundings} should
#'  be provided with a vector with the confounding names in the meta data (the
#'  meta data are provided via the parameter \code{anno}) as elements. As to
#'  \code{subset.CpGs}, it can also be the string "SCMER", and then the method
#'  \code{SCMER} will be used to select features from the data. In this case,
#'  other parameters should be set well, including \code{lasso}, \code{ridge},
#'  \code{n_pcs}, \code{perplexity}, \code{savefigures}, \code{pythonpath} and
#'  \code{topfeaturenumber}. Because \code{SCMER} is to select features able
#'  to preserve the original manifold structure of the data after its feature
#'  selection by elastic net, most of these parameters are used to config the
#'  manifold (\code{n_pcs} and \code{perplexity}) and elastic net processes
#'  (\code{lasso} and \code{ridge}), while another important parameter is the
#'  \code{pythonpath}, which is used to tell the function the absolute path of
#'  the \code{Python} interpreter you want to use to run \code{SCMER} because
#'  this method also depends on \code{Python}. Besides, because \code{limma}
#'  and \code{SCMER} can be time-consuming if running on large data, it is
#'  recommended to do a prescreen on the data before running them, and the
#'  parameter \code{topfeaturenumber} can be set as a numeric value such as
#'  50000, so that the top 50000 most variable features will be selected, and
#'  the top variable, \code{limma}, or the \code{SCMER} features can then be
#'  selected further on the prescreened data.
#'@param cores The core number need to do parallelization computation. Default
#'  is 10.
#'@param topfeaturenumber As mentioned in the \code{subset.CpGs} parameter
#'  part, it is used to set the prescreened feature number. Default is 50000.
#'  It can also be set as NULL, so that no precreen will be done on the data.
#'@param lasso A parameter special for \code{SCMER} feature selection and it
#'  defines the strength of L1 regularization in the elastic net process of
#'  \code{SCMER}. Default is 3.25e10-7, so that around 10000 features will be
#'  selected from 50000 prescreened candidate features.
#'@param ridge A parameter special for \code{SCMER} feature selection and it
#'  defines the strength of L2 regularization in the elastic net process of
#'  \code{SCMER}. Default is 0, so that the elastic net process is actually
#'  a LASSO process.
#'@param n_pcs Number of principle components need to reconstruct the sample-
#'  sample distance matrix during the \code{SCMER} selection. Default is 100.
#'@param perplexity Perplexity of tSNE modeling for the \code{SCMER} feature
#'  selection. Default is 10.
#'@param savefigures Whether save the PCA and UMAP figures generated by the
#'  \code{SCMER} method or not. Choose from TRUE and FALSE. Default is FALSE.
#'@param pythonpath Because the feature selection method \code{SCMER} is a
#'  \code{Python} based method, the directory of the \code{Python} interpreter
#'  you want to use to run it should be transferred to the function via this
#'  parameter, and several \code{Python} modules need to be installed to your
#'  \code{Python} environment, including \code{time}, \code{functiontools},
#'  \code{abc}, \code{torch}, \code{typing}, \code{sklearn}, \code{pandas},
#'  \code{numpy}, \code{matplotlib}, \code{multiprocessing}, \code{scanpy}.
#'@param anno A data frame recording the meta data of the samples, and should
#'  contain at least 2 columns named as "label" and "sentrix". The former one
#'  records the sample labels while the latter one records the sample IDs that
#'  also used as row names of the methylation data matrix. The default value
#'  is NULL and it is not necessary, but if need to use \code{limma} to do the
#'  feature selection and remove the confoundings, it should be provided with
#'  the confounding factors included in it.
#'@param confoundings A parameter special for \code{limma} feature selection.
#'  Details can be seen in the \code{subset.CpGs} parameter section.
#'@param padjcut A parameter for \code{limma} feature selection. Default value
#'  is 0.05 and details can be seen in the \code{subset.CpGs} section.
#'@param xcutoff A parameter for \code{limma}. Details can also be seen in the
#'  \code{subset.CpGs} section. Default value is 0.1.
#'@param cutnum A parameter special for \code{limma}. Details can be seen in
#'  the \code{subset.CpGs} section.
#'@return Will return a list containing the beta value matrix only with the
#'  selected features, and also containing a vector recording the names of the
#'  selected features.
#'@examples
#'library(methylClass)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'res <- mainfeature(betas.. = betas, subset.CpGs = 10000, cores = 4, 
#'  topfeaturenumber = 50000)
#'@export
mainfeature <- function(y.. = NULL,
                        betas.. = NULL,
                        betas.path,
                        subset.CpGs = 10000,

                        cores = 10,
                        topfeaturenumber = 50000,

                        #SCMER
                        lasso = 3.25e-7,
                        ridge = 0,
                        n_pcs = 100,
                        perplexity = 10,
                        savefigures = FALSE,
                        pythonpath = NULL,

                        #limma
                        anno = NULL,
                        confoundings = NULL,
                        padjcut = 0.05,
                        xcutoff = 0.1,
                        cutnum = 10000
){

  scmerpyfile <- system.file("python", "scmerpypackage.py", package = "methylClass")
  #scmerpyfile <- '/data/liuy47/nihcodes/scmerpypackage.py'

  if(!is.null(y..)){
    if(!is.factor(y..)){
      y.. <- as.character(y..)
      freqy <- table(y..)
      freqy <- freqy[order(-freqy)]
      y.. <- factor(y.., levels = names(freqy), ordered = TRUE)
    }
  }

  orianno <- anno

  if(is.null(betas..)){

    message("Loading betas\n")
    betas.. <- readRDS(betas.path)

  }

  if(!is.null(topfeaturenumber)){

    betas.. <- topvarfeatures(betasmat = betas..,
                              topfeaturenumber = topfeaturenumber)
  }

  betas <- betas..
  rm(betas..)

  if(is.numeric(subset.CpGs)){

    limit <- min(ncol(betas), subset.CpGs)

    betas <- topvarfeatures(betasmat = betas, topfeaturenumber = limit)

    betas <- betas[,colnames(betas)]

    features <- colnames(betas)


  }else if(subset.CpGs == 'SCMER'){

    if(!is.null(y..)){

      anno <- data.frame(label = as.character(y..),
                         sentrix = row.names(betas),
                         stringsAsFactors = FALSE)
      labeldat <- anno$label

    }else{

      anno <- NULL
      labeldat <- NULL

    }


    features <- scmerselection(trimbetasmat = betas,
                               annodat = anno,
                               labeldat = labeldat,
                               K = NULL,
                               k = NULL,
                               lasso = lasso,
                               ridge = ridge,
                               n_pcs = n_pcs,
                               perplexity = perplexity,
                               threads = cores,
                               savefigures = savefigures,
                               pythonpath = pythonpath,
                               scmerpyfile = scmerpyfile)

    message(paste0(length(features), ' SCMER probes were selected out of ',
                   ncol(betas), ' ones with lasso = ',
                   lasso, ' and ridge = ', ridge, '\n'))

    if(length(features) == 0){
      features <- colnames(betas)
    }


    betas <- betas[,features]

  }else if(subset.CpGs == 'limma' & (!is.null(y..) | !is.null(anno))){

    if(!is.null(y..) & is.null(anno)){

      anno <- data.frame(label = as.character(y..),
                         sentrix = row.names(betas),
                         stringsAsFactors = FALSE)

    }

    if(!('label' %in% colnames(anno))){

      if(!is.null(y..)){

        anno$label <- y..

      }else{

        return(NULL)

      }
    }

    #library(limma)

    vars <- unique(c('label', confoundings))
    vars <- intersect(vars, colnames(anno))
    vars <- c('label', setdiff(vars, 'label'))

    simpds <- anno[, vars, drop = FALSE]

    simpds <- simpds[complete.cases(simpds), , drop = FALSE]

    features <- diffsites(dat = t(betas), simpds = simpds,
                          padjcut = padjcut, xcutoff = xcutoff, cutnum = cutnum)

    message(paste0(length(features), ' limma probes were selected out of ',
                   ncol(betas), ' ones\n'))

    if(length(features) == 0){
      features <- colnames(betas)
    }

    betas <- betas[,features, drop = FALSE]

  }else{

    return(list(betas = betas,
                features = colnames(betas)))
  }

  res <- list(betas = betas,
              features = features)

  return(res)


}


#mainwrapper####


#'Cross validation on model performance
#'
#'Cross validation on model performance without calibration.
#'
#'@param y.. The true labels of the samples. Can be a vector, factor, or NULL.
#'  If it is a vector or factor, each element is a label for a sample and the
#'  element order in it should be the same as the sample order in the sample
#'  data provided by the parameter \code{betas..}. If it is NULL, there should
#'  be a vector or factor named `y` in the global environment and the function
#'  will load it and use it as the sample true labels.
#'@param betas.. The beta value matrix of the samples. Each row is one sample
#'  and each column is one feature. The function will divide the samples there
#'  internally into different cross validation training and testing sets to
#'  train and evaluate the model. This parameter can also be set as NULL, and
#'  in this case, the function will find and load the results of the function
#'  \code{cvdata}, which are the training and testing sets generated by it in
#'  advance. The absolute path of the folder with these divided data should be
#'  provided by the parameter \code{cv.betas.path}, and the name prefix of the
#'  data file should be provided by the parameter \code{cv.betas.prefix}, same
#'  as the value transferred to the parameter \code{out.fname} of the function
#'  \code{cvdata}.
#'@param cv.betas.path If the parameter \code{betas..} is set as NULL, this
#'  parameter is necessary to provide the folder path of the divided training
#'  and testing sets generated by the function \code{cvdata}, and it can be an
#'  absolute directory string or a relative one.
#'@param cv.betas.prefix If the parameter \code{betas..} is set as NULL, this
#'  parameter is necessary to provide the name prefix of the divided training
#'  and testing sets generated by the function \code{cvdata}, corresponding to
#'  the value of the parameter \code{out.fname} of the function \code{cvdata}.
#'@param subset.CpGs The feature selection method. It can be a numeric number
#'  such as 10000, and in this case, the top 10000 most variable features of
#'  the methylation data of each training set will be selected as the features
#'  to construct the machine learning model. It can also be the string "limma"
#'  and then \code{limma} will be used to select the differential features
#'  between a sample class and all other samples. The differential ones should
#'  fulfill the condition that the adjusted p-value < \code{padjcut} (default
#'  is 0.05), and its betas value difference between the sample class and all
#'  other samples should be > \code{xcutoff}, or < -\code{xcutoff} (default is
#'  0.1). After the differential features for each class have been found by
#'  \code{limma}, all of them will be mixed together and ordered according to
#'  the adjusted p-value and the absolute value of beta value difference, and
#'  finally the top \code{cutnum} features will be selected and used as the
#'  features (default value is 10000). Besides, if there are any confounding
#'  factors in the dataset need to be removed, they can be provided by the
#'  parameter \code{confoundings} and \code{limma} will select the features
#'  after these confoundings have been adjusted via linear regression. This
#'  \code{confoundings} should be provided as a vector with the names of the
#'  confoundings in the meta data (the meta data are provided by the parameter
#'  \code{anno}) as elements. As to \code{subset.CpGs}, it can also be set as
#'  the string "SCMER", and then \code{SCMER} will be used to select features
#'  from each training set. In this case, other parameters should be set well,
#'  including \code{lasso}, \code{ridge}, \code{n_pcs}, \code{perplexity},
#'  \code{savefigures}, \code{pythonpath}, \code{topfeaturenumber}. Because
#'  \code{SCMER} is a method to select features able to preserve the original
#'  manifold structure of the data after its feature selection via elastic net
#'  regularization, most of these parameters are used to config the manifold
#'  (\code{n_pcs} and \code{perplexity}) as well as elastic net (\code{lasso},
#'  \code{ridge}), while another important parameter is \code{pythonpath} that
#'  is used to tell the function the absolute path of \code{Python} you want
#'  to use to run \code{SCMER} because this method depends on \code{Python}.
#'  Also, because \code{limma} and \code{SCMER} can be time-consuming for a
#'  large number of candidate features, it is recommended to do a prescreen on
#'  the data before running them, and if the parameter \code{betas..} is not
#'  NULL, the parameter \code{topfeaturenumber} can be set as a numeric value
#'  such as 50000, so that before internally dividing the samples into the
#'  training and testing sets, the top 50000 most variable features will be
#'  selected on the whole dataset first, and then, after obtaining the cross
#'  validation training sets from it, the top variable, \code{limma}, or the
#'  \code{SCMER} features can be selected on these prescreened sets. However,
#'  if \code{betas..} is set as NULL and the training sets need to be loaded
#'  from the results of \code{cvdata}, \code{topfeaturenumber} will not work
#'  on them and the prescreened data need to be prepared during running the
#'  function \code{cvdata}. The parameters \code{subset.CpGs} can also be set
#'  as NULL, so that no feature selection will be done on the data before the
#'  model construction.
#'@param n.cv.folds A numeric number and the default value is 5, so that if
#'  the parameter \code{normalcv} is set as TRUE, a 5 fold cross validation
#'  will be performed, and if \code{normalcv} is FALSE, a 5 by 5 nested cross
#'  validation will be performed. This parameter only works if \code{betas..}
#'  is not NULL and an internal training/testing sets division is needed.
#'@param nfolds.. If the \code{n.cv.folds} parameter is provided as NULL, the
#'  training/testing sets need to be divided following the cross validation
#'  structure provided by this parameter and it is the result of the function
#'  \code{makecv}, but if it is also NULL, and there is such a data structure
#'  named "nfolds" in the environment, it will be loaded by the function and
#'  use it to divide the training/testing sets. This parameter only works when
#'  the \code{betas..} parameter is not NULL, because when it is NULL, this
#'  cross validation structure can be fetched directly from the data loaded by
#'  the parameters \code{cv.betas.path} and \code{cv.betas.prefix}. It is a
#'  list with the sample indexes in the dataset showing which samples belong
#'  to which training/testing sets.
#'@param K.start A number indicating from which outer cross validation loop
#'  the model construction and evaluation should be started. Default is 1.
#'@param k.start A number indicating from which inner cross validation loop
#'  the model construction and evaluation should be started. Default is 0, and
#'  when the cross validation is a normal one, rather than a nested one, this
#'  value should be 0.
#'@param K.stop The default of this value is NULL and indicates that the outer
#'  cross validation loops will be used to training and evaluate the model
#'  until all the loops have been finished. It can be also a number such as 3,
#'  so that the process will stop when the outer loop number has been 3.
#'@param k.stop The default of this value is NULL and indicates that the inner
#'  cross validation loops will be used to train and evaluate the model until
#'  all the loops have been finished. It can also be a number such as 3, so
#'  that this process will stop when the inner loop number has been 3, and for
#'  a normal cross validation without inner loops, this parameter should be
#'  NULL.
#'@param seed Some process performed by this function will need a seed number
#'  to fix the random process, such as the training/testing data division, the
#'  random sampling steps of some models such as random forest, eSVM, eNeural,
#'  etc, and this parameter is used to set their seeds. Default value is 1234.
#'@param out.path For all the cross validation loops, their result files will
#'  be saved in the folder set by this parameter. It is the folder name will
#'  be created in the current working directory, so a relative not absolute
#'  path.
#'@param out.fname The final cross validation result files will be saved in
#'  the folder set by the parameter \code{out.path} and the name prefix of the
#'  result files need to be set by this parameter, default is "CVfold".
#'@param cores The core number need to do parallelization computation. Default
#'  is 10.
#'@param topfeaturenumber As mentioned in the \code{subset.CpGs} parameter
#'  part, it is used to set the prescreened feature number when \code{betas..}
#'  is not NULL. Default value is 50000. It can also be set as NULL, so that
#'  no precreen will be done on the data.
#'@param normalcv Indicating whether the cross validation loops are normal or
#'  nested. Default is FALSE, meaning nested cross validation.
#'@param savefeaturenames Default is FALSE, but if is set as TRUE, the feature
#'  names selected by the feature selection process will be saved as a vector
#'  for each cross validation loop.
#'@param method Which algorithm need to be used to train the model. Can be a
#'  string as "RF", "SVM", "XGB", "ENet", "eNeural", "MOGONET", or "eSVM". The
#'  default value is "eSVM".
#'@param anno A data frame recording the meta data of the samples, and should
#'  contain at least 2 columns named as "label" and "sentrix". The former one
#'  records the sample labels while the latter one records the sample IDs that
#'  also used as row names of the methylation data matrices. The default value
#'  is NULL and it is not necessary as long as the \code{y..} parameter is
#'  provided, but if need to use \code{limma} to do the feature selection and
#'  remove the confounding factors, it should be provided with the confounding
#'  factors included in it.
#'@param lasso A parameter special for \code{SCMER} feature selection and it
#'  defines the strength of L1 regularization in the elastic net process of
#'  \code{SCMER}. Default is 3.25e10-7, so that around 10000 features will be
#'  selected from 50000 prescreened candidate features.
#'@param ridge A parameter special for \code{SCMER} feature selection and it
#'  defines the strength of L2 regularization in the elastic net process of
#'  \code{SCMER}. Default is 0, so that the elastic net process is actually
#'  a LASSO process.
#'@param n_pcs Number of principle components need to reconstruct the sample-
#'  sample distance matrix during the \code{SCMER} selection. Default is 100.
#'@param perplexity Perplexity of tSNE modeling for the \code{SCMER} feature
#'  selection. Default is 10.
#'@param savefigures Whether save the PCA and UMAP figures generated by the
#'  \code{SCMER} method or not. Choose from TRUE and FALSE. Default is FALSE.
#'@param pythonpath Because the feature selection method \code{SCMER} and the
#'  model training algorithm \code{MOGONET} are \code{Python} based methods,
#'  the directory of the \code{Python} interpreter you want to use to run them
#'  should be provided via this parameter, and to run \code{SCMER}, several
#'  modules should be installed to the \code{Python} environment, including
#'  \code{time}, \code{functiontools}, \code{abc}, \code{torch}, \code{numpy},
#'  \code{typing}, \code{pandas}, \code{matplotlib}, \code{multiprocessing},
#'  \code{scanpy}, and \code{sklearn}. To run \code{MOGONET}, the modules are
#'  \code{numpy}, \code{sklearn}, and \code{torch}.
#'@param confoundings A parameter special for \code{limma} feature selection.
#'  Details can be seen in the \code{subset.CpGs} parameter section.
#'@param padjcut A parameter for \code{limma} feature selection. Default value
#'  is 0.05 and details can be seen in the \code{subset.CpGs} section.
#'@param xcutoff A parameter for \code{limma}. Details can also be seen in the
#'  \code{subset.CpGs} section. Default value is 0.1.
#'@param cutnum A parameter special for \code{limma}. Details can be seen in
#'  the \code{subset.CpGs} section.
#'@param ntrees A parameter special for the random forest (RF) model, defining
#'  the number of decision trees in the RF model. Default is 500.
#'@param p A parameter special for RF. In the RF method here, a 2-step process
#'  is conducted. The first one constructs an RF model on all the candidate
#'  features with \code{ntrees} trees. Then, the top \code{p} most important
#'  features in this step will be selected by calculating their influence on
#'  the error of each tree using permuted out-of-bag data, and these features
#'  will be transferred to the second step to construct a second RF model on
#'  them, also with a tree number of \code{ntrees}. The parameter \code{p} is
#'  used to control how many top important features are needed to be selected,
#'  and the default value is 200.
#'@param modelcv For the models of SVM, eSVM, XGBoosting (XGB), elastic net
#'  (ENet) and eNeural, a hyperparameter search step is performed to find the
#'  optimal hyperparameters, via cross validation, such as the regularization
#'  constant of SVM, eSVM, and ENet, and this parameter is used to define the
#'  number of cross validation loops for hyperparameter search. Default is 5,
#'  and it means to train a model from the training set of each model training
#'  cross validation loop, this training set will be divided into 5 sets and
#'  a 5-fold cross validation will be used to evaluate the performance of the
#'  models with different hyperparameters and finally chooses the optimal one.
#'  Hence, a model training cross validation loop, such as the normal cross
#'  validation and nested cross validation loop, will further contain 5 cross
#'  validation loops for hyperparameter search.
#'@param C.base A parameter special for SVM and eSVM to set the regularization
#'  constant. This constant will be calculated by the function as base^index,
#'  and \code{C.base} here serves as the base number. Combined with other 2
#'  parameters \code{C.min} and \code{C.max} serving as indexes, it defines a
#'  regularization constant series. Its start is \code{C.base}^\code{C.min},
#'  and the end is \code{C.base}^\code{C.max}, while the near elements of the
#'  series have a difference of \code{C.base} fold. If the 2 indexes are set
#'  as the same, the series will become 1 regularization constant. The default
#'  value of \code{C.base} is 10.
#'@param C.min As mentioned in the \code{C.base} part, this parameter is used
#'  as the index of the small regularization constant number to set a series
#'  for SVM and eSVM. Default is -3.
#'@param C.max As mentioned in the \code{C.base} part, this parameter is used
#'  as the index of the large regularization constant number to set a series
#'  for SVM and eSVM. Default is -2.
#'@param learnernum A parameter special for eSVM, eNeural and \code{MOGONET}
#'  to set their base learner number. Default is 10.
#'@param minlearnersize A parameter special for eSVM, eNeural, \code{MOGONET}
#'  to define the lower limit of the feature number of their base learners.
#'  Default value is 1000, meaning each base learner should have at least 1000
#'  features after the random sampling process to sample features for them.
#'@param viewstandard When this parameter is set as NULL. The features will be
#'  assigned to the base learners of eSVM, eNeural and \code{MOGONET} through
#'  random sampling. While if it is "Relation_to_Island" and the features are
#'  DNA methylation probes, they will be split into groups of island probes,
#'  N shelf and N shore probes, S shelf and S shore probes, and opensea probes
#'  and then for each base learner, its features will be sampled from one of
#'  these groups. If this parameter is set as "UCSC_RefGene_Group", then the
#'  probes will be grouped into promoter probes, gene body probes and other
#'  probes and each base learner will get its features via sampling on one of
#'  these groups. The default value of this parameter is NULL.
#'@param platform When \code{viewstandard} is set as "Relation_to_Island" or
#'  "UCSC_RefGene_Group", this parameter will be used to define the platform
#'  of the probe annotation information to split them into different groups.
#'  The default value is "450K", and can also "EPIC".
#'@param max_depth A parameter special for XGB. Its the maximum depth of each
#'  tree. Default is 6.
#'@param eta A parameter special for XGB. It controls the learning rate via
#'  scaling the contribution of each tree by a factor of 0 < eta < 1 when the
#'  tree is added to the approximation, and can prevent overfitting by making
#'  the boosting process more conservative. Its default value is a vector of
#'  \code{c(0.1, 0.3)}, meaning a grid search will be conducted between these
#'  2 values to find the optimal \code{eta} value with less misclassification
#'  rate.
#'@param gamma A parameter special for XGB. Defines the minimum loss reduction
#'  required to make a further partition on a leaf node of the tree. Default
#'  value is a vector of \code{c(0, 0.01)} and a grid search will be conducted
#'  on it.
#'@param colsample_bytree Special for XGB. Defines subsample ratio of columns
#'  when constructing each tree. Default is \code{c(0.01, 0.02, 0.05, 0.2)},
#'  and a grid search will be performed on it.
#'@param subsample Subsample ratio of the training instance for XGB. Setting
#'  it to 0.5 means that XGB randomly collected half of the data instances to
#'  grow trees and this can prevent overfitting and make computation shorter.
#'  Its default value is 1.
#'@param min.chwght Minimum sum of instance weight (hessian) needed in a child
#'  of the tree in XGB. If the tree partition step results in a leaf node with
#'  the sum of instance weight less than this value, the building process will
#'  give up further partitioning. Default is 1.
#'@param nrounds A parameter special for XGB. Defines the max number of the
#'  boosting iterations. Default is 100.
#'@param early_stopping_rounds Special for XGB and default value is 50, which
#'  means the training with a validation set will stop if the performance dose
#'  not improve for 50 rounds.
#'@param alpha.min. A parameter special for ENet. Need to use with the other 2
#'  parameters \code{alpha.max.} and \code{by.} to set an elastic net mixing
#'  parameter series. The default value of \code{alpha.min.} is 0, the default
#'  value of \code{alpha.max.} is 1, and the default value of \code{by.} is
#'  0.1, so that a mixing parameter series staring with 0, ending with 1, and
#'  with a difference between its neighbor elements as 0.1 will be generated
#'  and to do a grid search on it to select the optimal mixing parameter value
#'  (alpha) giving the smallest MSE across the hyperparameter searching cross
#'  validation and then it will be used for the next model training.
#'@param alpha.max. A parameter special for ENet. Need to use with the other 2
#'  parameters \code{alpha.min.} and \code{by.} to set an elastic net mixing
#'  parameter series. As mentioned in the \code{alpha.min.} section.
#'@param by. A parameter special for ENet. Detail is in the \code{alpha.min.}
#'  section.
#'@param predefinedhidden A parameter special for eNeural. Use it to transfer
#'  the node number of each hidden layer of one neural network in the eNeural
#'  model. Need to be a vector, such as \code{c(100, 50)}, so that for each
#'  neural network in the eNeural model, 2 hidden layers will be set up. One
#'  is with 100 nodes, while the other is with 50 ones. Default value is NULL,
#'  so that the function will set up a hidden layer structure automatically.
#'  If the parameter \code{gridsearch} is set as FALSE, this structure is with
#'  2 layers and the node number of them are both around 1/100 of the input
#'  node number. If \code{gridsearch} is TRUE, several different hidden layer
#'  structures will be generated and a search will be performed on them to get
#'  the optimal one.
#'@param maxepchs A parameter special for eNeural and defines the epoch number
#'  for the neural network training. If the parameter \code{gridsearch} is set
#'  as FALSE, the epoch number is fixed as this, but if \code{gridsearch} is
#'  TRUE, an epoch number series will be set up starting from 10 and ending at
#'  \code{maxepchs}, with the neighbor elements having a 10 fold difference.
#'  Then, grid search will be performed across this series. The default value
#'  of \code{maxepchs} is 10.
#'@param activation Activation function special for eNeural. Can be a string
#'  or a vector of strings to do grid search. Default is "Rectifier". Can also
#'  be "Tanh" or "Maxout", or a vector with elements from them.
#'@param momentum_start Special for eNeural. Defines the initial momentum at
#'  the beginning of training (try 0.5). Default is 0. And a vector covering
#'  different values can be used for hyperparameter grid search.
#'@param rho Speical parameter for eNeural. Adaptive learning rate time decay
#'  factor (defines the similarity to prior updates). Default is 0.99. Can be
#'  a vector for hyperparameter grid search.
#'@param gridsearch Special parameter for eNeural. Whether do grid search to
#'  select the optimal hyperparameters, or directly use the fixed and given
#'  hyperparameters to train the neural networks. If it is TRUE, grid search
#'  will be performed on the hyperparameters of hidden layer size and depth,
#'  epoch number, activation function, initial momentum, and adaptive learning
#'  rate time decay factor.
#'@param num_epoch_pretrain Special parameter for \code{MOGONET} and defines
#'  the epoch number for its pretraining process. Default is 500.
#'@param num_epoch Special parameter for \code{MOGONET} and defines the epoch
#'  number for its training process. Default is 2500.
#'@param adj_parameter Special parameter for \code{MOGONET} and defines the
#'  the average number of edges per node that are retained in the adjacency
#'  matrix used for graph convolutional networks (GCNs) construction. Default
#'  is 10.
#'@param dim_he_list Special parameter for \code{MOGONET} and is to define the
#'  node number of each hidden layer of the GCN network. Need to be a vector
#'  with numbers as elements, such as \code{c(400, 400, 200)}, so that in each
#'  GCN networks in \code{MOGONET}, 3 hidden layers will be set up. One is with
#'  400 nodes, while the others are with 400 and 200 nodes. The default value
#'  is \code{c(400, 400, 200)}.
#'@param lr_e_pretrain Special parameter for \code{MOGONET} and used to define
#'  the learning rate of the GCN networks for the single-omics data at their
#'  pretraining stage. Default value is 1e-3.
#'@param lr_e Special parameter for \code{MOGONET} and defines the learning
#'  rate of the GCN networks for the single-omics data at the training stage.
#'  Default value is 5e-4.
#'@param lr_c Special parameter for \code{MOGONET} and defines the learning
#'  rate of the view correlation discovery network (VCDN) to aggregate the GCN
#'  network results. Default is 1e-3.
#'@param multiomicsnames Used for multi-omics model training. In this case, a
#'  matrix should be organized using rows as samples and columns as features,
#'  and the features should come from all the omics data want to use. Then,
#'  same as the methylation data, the function can receive the matrix via the
#'  parameter \code{betas..} or via loading the training/testing sets division
#'  results of the function \code{cvdata}. To demonstrate which features in
#'  the matrix are from which omics, the parameter \code{multiomicsnames} need
#'  to be used to transfer a vector to the function. The element order in the
#'  vector should be the same as the feature order in the matrix. An element
#'  is the omics names of one feature. The default value is NULL and in this
#'  case the data will be treated as single-omics data, but if an omics name
#'  indication vector is provided, the data will be treated as multi-omics.
#'@return The cross validation results will be saved in the directory set by
#'  \code{out.path}, and each cross validation loop will have an .RData file
#'  saved there, with the testing sample prediction score matrix, the trained
#'  model object, the training and testing sample indexes of that loop, etc in
#'  it. These results are the results from the raw model and are needed by the
#'  calibration step conducted by the function \code{maincalibration}.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'maincv(y.. = labels, betas.. = betas, subset.CpGs = 10000, n.cv.folds = 5, 
#'  normalcv = TRUE, out.path = 'RFCV', out.fname = 'CVfold', 
#'  method = 'RF', seed = 1234, cores = 4)
#'  
#'\dontrun{
#'maincv(y.. = labels, betas.. = betas, subset.CpGs = 10000, n.cv.folds = 5, 
#'  normalcv = TRUE, out.path = 'eSVMCV', out.fname = 'CVfold', 
#'  method = 'eSVM', seed = 1234, cores = 4)
#'}
#'@export
maincv <- function(y.. = NULL,
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

                   #SCMER & limma
                   anno = NULL,

                   #SCMER
                   lasso = 3.25e-7,
                   ridge = 0,
                   n_pcs = 100,
                   perplexity = 10,
                   savefigures = FALSE,
                   pythonpath = NULL,

                   #limma
                   confoundings = NULL,
                   padjcut = 0.05,
                   xcutoff = 0.1,
                   cutnum = 10000,

                   #RF
                   ntrees = 500,
                   p = 200,

                   #SVM, eSVM, XGBoosting, GLMNET, eNeural
                   modelcv = 5,

                   #SVM, eSVM
                   C.base = 10,
                   C.min = -3,
                   C.max = -2,

                   #eSVM, eNeural, MOGONET
                   learnernum = 10,
                   minlearnersize = 1000,
                   viewstandard = NULL,
                   platform = "450K",

                   #XGBoosting
                   max_depth = 6,
                   eta = c(0.1, 0.3),
                   gamma = c(0, 0.01),
                   colsample_bytree = c(0.01, 0.02, 0.05, 0.2),
                   subsample = 1,
                   min.chwght = 1,
                   nrounds = 100, # use default to limit computational burden
                   early_stopping_rounds = 50,

                   #GLMNET
                   alpha.min. = 0,
                   alpha.max. = 1,
                   by. = 0.1,

                   #eNeural
                   predefinedhidden = NULL,
                   maxepochs = 10,

                   activation = "Rectifier",

                   momentum_start = 0,
                   rho = 0.99,

                   gridsearch = FALSE,

                   #MOGONET
                   num_epoch_pretrain = 500,
                   num_epoch = 2500,
                   adj_parameter = 10,
                   dim_he_list = c(400, 400, 200),

                   lr_e_pretrain = 1e-3,
                   lr_e = 5e-4,
                   lr_c = 1e-3,

                   multiomicsnames = NULL,
                   weighted = FALSE){

  trainsub <- 'betas.train'
  testsub <- 'betas.test'
  n.rep. <- 1

  #scmerpyfile <- system.file("python", "scmerpypackage.py", package = "methylClass")
  scmerpyfile <- '/data/liuy47/nihcodes/scmerpypackage.py'

  if(gridsearch == FALSE){
    activation <- activation[1]
    momentum_start <- momentum_start[1]
    rho <- rho[1]
  }

  #mogonetpyfile <- system.file("python", "mogonet_r.py", package = "methylClass")
  mogonetpyfile <- '/data/liuy47/nihcodes/mogonet_r.py'

  test_inverval <- 50
  featurescale <- TRUE

  if(platform == 'EPIC'){
    platform <- 850
  }else if(platform == '450K'){
    platform <- 450
  }



  if(!is.null(multiomicsnames)){
    subset.CpGs <- NULL
    topfeaturenumber <- NULL
  }

  orianno <- anno

  if(is.null(y..)){

    if(exists("y")){
      y.. <- get("y", envir = .GlobalEnv)
      message("`y` label was fetched from .GlobalEnv\n")
    }else{
      stop("Please provide `y..` labels\n")
    }

  }

  if(!is.factor(y..)){
    y.. <- as.character(y..)
    freqy <- table(y..)
    freqy <- freqy[order(-freqy)]
    y.. <- factor(y.., levels = names(freqy), ordered = TRUE)

  }

  if(!is.null(n.cv.folds) && !is.null(betas..)){

    nfolds.. <- makecv(y = y.., cv.fold = n.cv.folds, seednum = seed,
                       normalcv = normalcv)

  }else if(is.null(nfolds..) && exists("nfolds") && !is.null(betas..)){

    nfolds.. <- get("nfolds", envir = .GlobalEnv)
    message("`nfolds` nested CV scheme assignment was fetched from .GlobalEnv\n")

    n.cv.folds <- length(nfolds..)

  }else if(is.null(betas..)){

    if(!is.null(nfolds..)){
      rm(nfolds..)
    }

    if(!is.null(n.cv.folds)){
      rm(n.cv.folds)
    }

    path2load <- file.path(cv.betas.path)

    cvfiles <- dir(path2load)
    cvfiles <- cvfiles[grepl(pattern=paste0(cv.betas.prefix, '.*\\.RData'), x = cvfiles)]

    ks <- gsub(pattern = '\\.RData', replacement = '', x = cvfiles)
    ks <- gsub(pattern = '^.*\\.', replacement = '', x = ks)
    kmax <- max(as.numeric(ks))

    Ks <- gsub(pattern = '\\.RData', replacement = '', x = cvfiles)
    Ks <- unlist(lapply(X = strsplit(x = Ks, split = '\\.', fixed = FALSE),
                        FUN = function(x){x[length(x) - 1]}))
    Kmax <- max(as.numeric(Ks))

    if(kmax == 0){
      normalcv <- TRUE
      n.cv.folds <- Kmax
    }else{
      normalcv <- FALSE
      n.cv.folds <- min(Kmax, kmax)
    }


  }else if(!is.null(betas..)){
    stop("Please provide a fold structure or the CV number for resampling\n")
  }

  #print(paste0('nfolds.. = ', nfolds..))

  if(!is.null(betas..)){

    if(length(y..) == ncol(betas..) & length(y..) != nrow(betas..)){
      betas.. <- t(betas..)
    }

  }

  if(!is.null(topfeaturenumber)){

    if(is.null(betas..)){

      topfeaturenumber <- NULL

    }else{

      betas.. <- topvarfeatures(betasmat = betas..,
                                topfeaturenumber = topfeaturenumber)

    }

  }

  if(is.null(K.stop) && is.null(k.stop)) {

    K.stop <- n.cv.outer.folds <- n.cv.folds
    k.stop <- n.cv.inner.folds <- n.cv.folds

  }else{

    n.cv.outer.folds <-  K.stop
    n.cv.inner.folds <- k.stop

  }

  # Outer loop
  K <- 1
  for(K in K.start:n.cv.outer.folds){
    cat(paste0('K = ', K, '\n'))
    # Schedule/Stop nested CV
    ncv.scheduler  <- subfunc_nestedcv_scheduler(K = K,
                                                 K.start = K.start,
                                                 K.stop = K.stop,
                                                 k.start = k.start,
                                                 k.stop = k.stop,
                                                 n.cv.folds = n.cv.folds,
                                                 n.cv.inner.folds = n.cv.inner.folds)
    k.start <- ncv.scheduler$k.start
    n.cv.inner.folds <- ncv.scheduler$n.cv.inner.folds

    # Inner/Nested loop
    k <- 0
    for(k in k.start:n.cv.inner.folds){
      cat(paste0('k = ', k, '\n'))

      if(k > 0 & normalcv == FALSE & !is.null(betas..)){

        message("Calculating inner/nested fold ", K,".", k,"  ... ",Sys.time())
        fold <- nfolds..[[K]][[2]][[k]]

      }else if(k > 0 & normalcv == TRUE & !is.null(betas..)){

        next()

      }else if(!is.null(betas..)){

        message("Calculating outer fold ", K, ".0  ... ", Sys.time())
        fold <- nfolds..[[K]][[1]][[1]]

      }

      # 1. Load `betas..`
      if(is.null(betas..)){

        message("Loading betas for (sub)fold ", K, ".", k, '\n')
        # Safe loading into separate env
        env2load <- environment()

        fname2load <- file.path(path2load, paste(cv.betas.prefix, K, k, "RData", sep = "."))
        # Load into env
        load(file = fname2load, envir = env2load)
        # Get
        betas.train <- get(x = trainsub, envir = env2load)
        betas.test <- get(x = testsub, envir = env2load)
        #Note that betas.train and betas.test columns/CpGs are ordered in deacreasing = T
        fold <- get(x = 'fold', envir = env2load)

        if(is.null(subset.CpGs)){

          betas.test <- betas.test[,colnames(betas.train)]

        }else{

          if(is.null(orianno)){

            anno <- data.frame(label = as.character(y..[fold$train]),
                               sentrix = row.names(betas.train),
                               stringsAsFactors = FALSE)
          }else if(!('label' %in% colnames(orianno))){

            anno <- data.frame(label = as.character(y..[fold$train]),
                               sentrix = row.names(betas.train),
                               stringsAsFactors = FALSE)

          }else{

            if('sentrix' %in% colnames(orianno)){

              if(sum(!(orianno$sentrix %in% row.names(betas.train))) == 0){

                sharedsamples <- intersect(orianno$sentrix, row.names(betas.train))
                anno <- subset(orianno, sentrix %in% sharedsamples)
                betas.train <- betas.train[anno$sentrix, , drop = FALSE]

              }else if(sum(!(row.names(orianno) %in% row.names(betas.train))) == 0){

                sharedsamples <- intersect(row.names(orianno), row.names(betas.train))
                anno <- orianno[sharedsamples, , drop = FALSE]
                betas.train <- betas.train[row.names(anno), , drop = FALSE]

              }else{

                anno <- data.frame(label = as.character(y..[fold$train]),
                                   sentrix = row.names(betas.train),
                                   stringsAsFactors = FALSE)

              }

            }else if(sum(!(row.names(orianno) %in% row.names(betas.train))) == 0){

              sharedsamples <- intersect(row.names(orianno), row.names(betas.train))
              anno <- orianno[sharedsamples, , drop = FALSE]
              betas.train <- betas.train[row.names(anno), , drop = FALSE]

            }else{

              anno <- data.frame(label = as.character(y..[fold$train]),
                                 sentrix = row.names(betas.train),
                                 stringsAsFactors = FALSE)

            }



          }

          features <- mainfeature(y.. = as.character(y..[fold$train]),
                                  betas.. = betas.train,
                                  subset.CpGs = subset.CpGs,

                                  cores = cores,
                                  topfeaturenumber = NULL,

                                  #SCMER
                                  lasso = lasso,
                                  ridge = ridge,
                                  n_pcs = n_pcs,
                                  perplexity = perplexity,
                                  savefigures = savefigures,
                                  pythonpath = pythonpath,

                                  #limma
                                  anno = anno,
                                  confoundings = confoundings,
                                  padjcut = padjcut,
                                  xcutoff = xcutoff,
                                  cutnum = cutnum)

          features <- features$features

          if(savefeaturenames == TRUE){

            filetag <- Sys.time()
            filetag <- gsub(pattern = ':', replacement = '-', x = filetag)
            filetag <- gsub(pattern = ' ', replacement = '-', x = filetag)

            folder.path <- file.path(getwd(), out.path, 'features')
            if(!dir.exists(folder.path)){
              dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
            }

            featureprefix <- 'probes'
            if(is.numeric(subset.CpGs)){
              featureprefix <- paste0('Top.', subset.CpGs, '.varprobes.')
            }else if(subset.CpGs == 'SCMER'){
              featureprefix <- 'SCMERprobes.'
            }else if(subset.CpGs == 'limma'){
              featureprefix <- 'limmaprobes.'
            }


            save(features,
                 file = file.path(folder.path,
                                  paste0(featureprefix, K, '.', k, '.', filetag, '.RData'))
            )

          }

          betas.train <- betas.train[,features]
          betas.test <- betas.test[,features]

        }

      }else{

        betas.train <- betas..[fold$train,]
        betas.test <- betas..[fold$test,]

        if(is.null(subset.CpGs)){

          betas.test <- betas.test[,colnames(betas.train)]

        }else{

          if(is.null(orianno)){

            anno <- data.frame(label = as.character(y..[fold$train]),
                               sentrix = row.names(betas.train),
                               stringsAsFactors = FALSE)
          }else if(!('label' %in% colnames(orianno))){

            anno <- data.frame(label = as.character(y..[fold$train]),
                               sentrix = row.names(betas.train),
                               stringsAsFactors = FALSE)

          }else{

            if('sentrix' %in% colnames(orianno)){

              if(sum(!(orianno$sentrix %in% row.names(betas.train))) == 0){

                sharedsamples <- intersect(orianno$sentrix, row.names(betas.train))
                anno <- subset(orianno, sentrix %in% sharedsamples)
                betas.train <- betas.train[anno$sentrix, , drop = FALSE]

              }else if(sum(!(row.names(orianno) %in% row.names(betas.train))) == 0){

                sharedsamples <- intersect(row.names(orianno), row.names(betas.train))
                anno <- orianno[sharedsamples, , drop = FALSE]
                betas.train <- betas.train[row.names(anno), , drop = FALSE]

              }else{

                anno <- data.frame(label = as.character(y..[fold$train]),
                                   sentrix = row.names(betas.train),
                                   stringsAsFactors = FALSE)

              }

            }else if(sum(!(row.names(orianno) %in% row.names(betas.train))) == 0){

              sharedsamples <- intersect(row.names(orianno), row.names(betas.train))
              anno <- orianno[sharedsamples, , drop = FALSE]
              betas.train <- betas.train[row.names(anno), , drop = FALSE]

            }else{

              anno <- data.frame(label = as.character(y..[fold$train]),
                                 sentrix = row.names(betas.train),
                                 stringsAsFactors = FALSE)

            }



          }

          features <- mainfeature(y.. = as.character(y..[fold$train]),
                                  betas.. = betas.train,
                                  subset.CpGs = subset.CpGs,

                                  cores = cores,
                                  topfeaturenumber = NULL,

                                  #SCMER
                                  lasso = lasso,
                                  ridge = ridge,
                                  n_pcs = n_pcs,
                                  perplexity = perplexity,
                                  savefigures = savefigures,
                                  pythonpath = pythonpath,

                                  #limma
                                  anno = anno,
                                  confoundings = confoundings,
                                  padjcut = padjcut,
                                  xcutoff = xcutoff,
                                  cutnum = cutnum)










          features <- features$features

          if(savefeaturenames == TRUE){

            filetag <- Sys.time()
            filetag <- gsub(pattern = ':', replacement = '-', x = filetag)
            filetag <- gsub(pattern = ' ', replacement = '-', x = filetag)

            folder.path <- file.path(getwd(), out.path, 'features')
            if(!dir.exists(folder.path)){
              dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
            }

            featureprefix <- 'probes'
            if(is.numeric(subset.CpGs)){
              featureprefix <- paste0('Top.', subset.CpGs, '.varprobes.')
            }else if(subset.CpGs == 'SCMER'){
              featureprefix <- 'SCMERprobes.'
            }else if(subset.CpGs == 'limma'){
              featureprefix <- 'limmaprobes.'
            }


            save(features,
                 file = file.path(folder.path,
                                  paste0(featureprefix, K, '.', k, '.', filetag, '.RData'))
            )

          }

          betas.train <- betas.train[,features]
          betas.test <- betas.test[,features]

        }
      }

      #Tune & train on training set


      if(method == 'RF'){

        # Check
        if(max(p) > ncol(betas.train)) { stop("< Error >: maximum value of `p` [",
                                              max(p), "] is larger than available in `betas.train`: [",
                                              ncol(betas.train), "]. Please adjust the function call\n")}

        message("Start tuning on training set using RF ... @ ", Sys.time(), "\n")

        rfcv <- trainRF(y = y..[fold$train],
                        betas = betas.train,
                        ntrees = ntrees,
                        p = p,
                        seed = seed,
                        cores = cores)

        message("Fit tuned RF on: test set",
                "(n_cases: ", nrow(betas.test), "): ", K, ".", k, " ... @ ",
                Sys.time(), "\n")

        scores <- predict(rfcv[[1]],
                          betas.test[, match(rownames(rfcv[[1]]$importance),
                                             colnames(betas.test))],
                          type = "prob")

        #Calculate Misclassification Errors (ME) on test set
        err <- sum(colnames(scores)[apply(scores, 1, which.max)] != y..[fold$test])/length(fold$test)
        message("Misclassification error: ", err, " @ ", Sys.time(), "\n")

        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')

        folder.path <- file.path(getwd(), out.path)
        if(!dir.exists(folder.path)){
          dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
        }

        save(scores, rfcv, fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )


      }else if(method == 'SVM'){

        message("Start tuning on training set using SVM ... @ ", Sys.time(), "\n")

        svm.linearcv <- train_SVM_e1071_LK(y = y..[fold$train],
                                           betas.Train = betas.train,
                                           seed = seed + 1,
                                           nfolds = modelcv,
                                           mc.cores = cores,
                                           C.base = C.base,
                                           C.min = C.min,
                                           C.max = C.max,
                                           weighted = weighted)

        message("Predict SVM model with tuned cost (C) parameter ... ",
                "\n Note: \'If the training set was scaled by svm (done by default),",
                " the new data is scaled accordingly using scale and center of the training data.\' ... @ ",
                Sys.time(), '\n')

        scores.pred.svm.e1071.obj <- predict(object = svm.linearcv[[1]],
                                             newdata = betas.test,
                                             probability = TRUE,
                                             decisionValues = TRUE)

        # probs.pred.SVM.e1071.obj => is a factor with attributes
        # Get probabilities
        scores.pred.svm.e1071.mtx <- attr(scores.pred.svm.e1071.obj, "probabilities")
        # !!!CAVE: colnames() order might not be the same as in levels(y) originally!!!

        err.svm.e1071.probs <- sum(colnames(scores.pred.svm.e1071.mtx)[apply(scores.pred.svm.e1071.mtx, 1, which.max)] != y..[fold$test])/length(fold$test)
        message("Misclassification error on test set estimated using [probabilities matrix] output: ",
                err.svm.e1071.probs, " ; @ ", Sys.time(), '\n')

        # Control Steps
        scores.pred.svm.e1071.mtx <- scores.pred.svm.e1071.mtx[rownames(betas.test), , drop = FALSE]
        scores.pred.svm.e1071.mtx <- scores.pred.svm.e1071.mtx[, levels(y..[fold$test]), drop = FALSE]


        message("\nControl step: whether rownames are identical:",
                identical(rownames(scores.pred.svm.e1071.mtx), rownames(betas.test)), '\n')
        message("Control step: whether colnames are identical:",
                identical(colnames(scores.pred.svm.e1071.mtx), levels(y..)), '\n')

        if(identical(colnames(scores.pred.svm.e1071.mtx), levels(y..[fold$test])) == FALSE){
          message("CAVE: Order of levels(y) and colnames(probs.pred.SVM.e1071.mtx)",
                  " => needs matching during performance evaluation!\n")
        }

        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')

        folder.path <- file.path(getwd(), out.path)
        if(!dir.exists(folder.path)){
          dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
        }

        save(scores.pred.svm.e1071.mtx,
             scores.pred.svm.e1071.obj,
             svm.linearcv,
             fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )

      }else if(method == 'XGB'){

        message("Data preprocessing: reformatting [betas] as xgb.DMatrix & [y] outcome as numeric-1 ... @ ",
                Sys.time(), '\n')

        y.num <- as.numeric(y..)
        y.xgb.train <- y.num[fold$train]-1
        train.K.k.mtx <- betas.train

        xgb.train.K.k.dgC <- tryCatch({

          as(betas.train, "CsparseMatrix") #sparse mtx (supposed to be faster)

        }, error = function(err){

          Matrix::Matrix(betas.train, sparse = TRUE)

        })

        xgb.train.K.k.dgC <- as(betas.train, "CsparseMatrix")  #sparse mtx (supposed to be faster)
        xgb.train.K.k.l <- list(data = xgb.train.K.k.dgC, label = y.xgb.train)
        dtrain <- xgboost::xgb.DMatrix(data = xgb.train.K.k.l$data,
                                       label = xgb.train.K.k.l$label) #xgboost own data structure

        y.xgb.test <- y.num[fold$test]-1
        test.K.k.mtx <- betas.test
        xgb.test.K.k.dgC <- as(betas.test, "CsparseMatrix")
        xgb.test.K.k.l <- list(data = xgb.test.K.k.dgC, label = y.xgb.test)
        dtest <- xgboost::xgb.DMatrix(data = xgb.test.K.k.l$data,
                                      label = xgb.test.K.k.l$label)

        watchlist <- list(train = dtrain, test = dtest)

        message("Start xgboost-CARET tuning on training set using XGBoosting ... @ ", Sys.time(), '\n')
        xgb.train.fit.caret <- trainXGBOOST_caret_tuner(y = y..[fold$train], #factor-caret does not need numeric conversion
                                                        train.K.k.mtx = train.K.k.mtx,
                                                        K. = K,
                                                        k. = k,
                                                        dtrain. = dtrain,
                                                        watchlist. = watchlist,
                                                        n.CV = modelcv,
                                                        n.rep = n.rep.,
                                                        seed. = seed,
                                                        allow.parallel = TRUE,
                                                        mc.cores = cores,
                                                        max_depth. = max_depth,
                                                        eta. = eta,
                                                        gamma. = gamma,
                                                        colsample_bytree. = colsample_bytree,
                                                        subsample. = subsample,
                                                        minchwght = min.chwght,
                                                        early_stopping_rounds. = early_stopping_rounds,
                                                        nrounds. = nrounds,
                                                        objective. = "multi:softprob",
                                                        eval_metric. = "merror",
                                                        save.xgb.model = TRUE)

        message("Predict on test set using best xgboosted tree model ... @ ", Sys.time(), '\n')

        scores.pred.xgboost.vec.test <- predict(object = xgb.train.fit.caret[[1]],
                                                newdata = test.K.k.mtx,
                                                ntreelimit = xgb.train.fit.caret[[1]]$best_iteration,
                                                outputmargin = F)

        scores.pred.xgboost <- matrix(scores.pred.xgboost.vec.test,
                                      nrow = nrow(test.K.k.mtx),
                                      ncol = length(levels(y..)),
                                      byrow = T)

        rownames(scores.pred.xgboost) <- rownames(betas.test)
        colnames(scores.pred.xgboost) <- levels(y..)

        err.xgb.K.k <- sum(colnames(scores.pred.xgboost)[apply(scores.pred.xgboost, 1, which.max)] != y..[fold$test])/length(fold$test)
        message("Misclassification Error on test set ", K, ".", k, " = ", err.xgb.K.k, '\n')

        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')

        folder.path <- file.path(getwd(), out.path)
        dir.create(folder.path, recursive = TRUE, showWarnings = FALSE)

        save(scores.pred.xgboost,
             xgb.train.fit.caret,
             fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )

      }else if(method == 'ENet'){

        message("Start tuning on training set using ElasticNet ... @ ", Sys.time(), '\n')

        glmnetcv.tuned <- trainGLMNET(y = y..[fold$train],
                                      betas = betas.train,
                                      seed = seed,
                                      alpha.min = alpha.min.,
                                      alpha.max = alpha.max.,
                                      by = by.,
                                      nfolds.cvglmnet = modelcv,
                                      mc.cores = cores)

        message("Fit tuned glmnet on: test set ", K, ".", k, " ... @ ", Sys.time(), '\n')

        probs <- predict(glmnetcv.tuned[[1]],
                         newx = betas.test,
                         type="response")[,,1]

        err.probs.glmnet <- sum(colnames(probs)[apply(probs, 1, which.max)] != y..[fold$test])/length(fold$test)

        message("Misclassification error on [Test Set] CVfold.", K, ".", k,
                "\n of ElasticNet with alpha = ",
                glmnetcv.tuned[[3]]$opt.alpha,
                " and lambda = ",
                glmnetcv.tuned[[3]]$opt.lambda,
                " setting is: ",
                err.probs.glmnet,
                " @ ", Sys.time(), '\n')


        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')

        folder.path <- file.path(getwd(), out.path)
        dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)

        save(probs,
             glmnetcv.tuned,
             fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )



      }else if(method == 'eNeural'){

        message("Start tuning on training set using eNeural ... @ ", Sys.time(), '\n')

        eNeuralcv <- eNeural(x = betas.train,
                             y = y..[fold$train],
                             cross = modelcv,
                             seednum = seed,
                             samplenum = learnernum,
                             cores = cores,

                             predefinedhidden = predefinedhidden,
                             maxepochs = maxepochs,

                             activation = activation,

                             momentum_start = momentum_start,
                             rho = rho,

                             gridsearch = gridsearch,
                             savefile = FALSE,

                             minsize = minlearnersize,
                             viewstandard = viewstandard,
                             platform = platform,
                             multiomicsnames = multiomicsnames)

        message("Predict eNeural model ... @ ",
                Sys.time(), '\n')

        scores.pred.eNeural <- eNeuralpredict(eNeuralmod = eNeuralcv[[1]],
                                              x = betas.test,
                                              cores = cores)

        err.eNeural.probs <- sum(colnames(scores.pred.eNeural)[apply(scores.pred.eNeural, 1, which.max)] !=
                                   y..[fold$test]) / length(fold$test)
        message("Misclassification error on test set estimated using [probabilities matrix] output: ",
                err.eNeural.probs, " ; @ ", Sys.time(), '\n')

        message("Control step: whether rownames are identical:",
                identical(rownames(scores.pred.eNeural), rownames(betas.test)), '\n')
        message("Control step: whether colnames are identical:",
                identical(colnames(scores.pred.eNeural), levels(y..)), '\n')



        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')

        folder.path <- file.path(getwd(), out.path)
        if(!dir.exists(folder.path)){
          dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
        }

        save(scores.pred.eNeural,
             eNeuralcv,
             fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )

        baselearnerpath <- file.path(folder.path, paste(out.fname, K, k, 'baselearners', sep = '.'))

        if(!dir.exists(baselearnerpath)){
          dir.create(baselearnerpath, showWarnings = FALSE, recursive = TRUE)
        }

        for(i in 1:length(eNeuralcv[[1]]$baselearners)){

          baselearner <- eNeuralcv[[1]]$baselearners[[i]]

          h2o::h2o.saveModel(baselearner,
                             path = baselearnerpath,
                             force = TRUE)


        }



      }else if(method == 'MOGONET'){

        message("Start tuning on training set using MOGOnet ... ", Sys.time(), '\n')

        probesplitres <- probesplit(traindat = betas.train,
                                    testdat = betas.test,
                                    samplenum = learnernum,
                                    seednum = seed,
                                    minsize = minlearnersize,
                                    viewstandard = viewstandard,
                                    platform = platform,
                                    multiomicsnames = multiomicsnames)

        scoreses <- mogonet(traingroupdatlist = probesplitres$traingroupdatlist,
                            testgroupdatlist = probesplitres$testgroupdatlist,
                            trainlabels = y..[fold$train],
                            testlabels = y..[fold$test],

                            pythonpath = pythonpath,
                            mogonetpyfile = mogonetpyfile,
                            K = K,
                            k = k,

                            num_epoch_pretrain = num_epoch_pretrain,
                            num_epoch = num_epoch,
                            adj_parameter = adj_parameter,
                            dim_he_list = dim_he_list,

                            lr_e_pretrain = lr_e_pretrain,
                            lr_e = lr_e,
                            lr_c = lr_c,
                            seednum = seed,
                            test_inverval = test_inverval)

        scores <- scoreses$testres

        err <- sum(colnames(scores)[apply(scores, 1, which.max)] != y..[fold$test]) / length(fold$test)

        message("Misclassification error on test set estimated using [probabilities matrix] output: ",
                err, " ; @ ", Sys.time(), '\n')
        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')



        folder.path <- file.path(getwd(), out.path)
        if(!dir.exists(folder.path)){
          dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
        }

        save(scores, fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )

      }else{

        message("Start tuning on training set using eSVM ... @ ", Sys.time(), '\n')

        svm.linearcv <- train_eSVM(y = y..[fold$train],
                                   betas.Train = betas.train,
                                   seed = seed + 1,
                                   mc.cores = cores,
                                   nfolds = modelcv,
                                   C.base = C.base,
                                   C.min = C.min,
                                   C.max = C.max,
                                   samplenum = learnernum,
                                   mod.type = "C-classification",
                                   minsize = minlearnersize,
                                   viewstandard = viewstandard,
                                   platform = platform,
                                   multiomicsnames = multiomicsnames,
                                   featurescale = featurescale,
                                   weighted = weighted)

        message("Predict eSVM model with tuned cost (C) parameter ... ",
                "\n Note: If the training set was scaled by eSVM (done by default),",
                " the new data is scaled accordingly using scale and center of the training data.\' ... @ ",
                Sys.time(), '\n')

        scores.pred.eSVM <- eSVMpredict(eSVMmod = svm.linearcv[[1]],
                                        x = betas.test,
                                        cores = cores)

        err.eSVM.probs <- sum(colnames(scores.pred.eSVM)[apply(scores.pred.eSVM, 1, which.max)] !=
                                y..[fold$test]) / length(fold$test)
        message("Misclassification error on test set estimated using [probabilities matrix] output: ",
                err.eSVM.probs, " ; @ ", Sys.time(), '\n')

        message("Control step: whether rownames are identical:",
                identical(rownames(scores.pred.eSVM), rownames(betas.test)), '\n')
        message("Control step: whether colnames are identical:",
                identical(colnames(scores.pred.eSVM), levels(y..)), '\n')



        message("Saving output objects & creating output folder (if necessary) @ ", Sys.time(), '\n')

        folder.path <- file.path(getwd(), out.path)
        if(!dir.exists(folder.path)){
          dir.create(folder.path, showWarnings = FALSE, recursive = TRUE)
        }

        save(scores.pred.eSVM,
             svm.linearcv,
             fold,
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
        )

      }

    }
  }

  message("Full run finished @ ", Sys.time(), '\n')
}


#calibration

sub_performance_evaluator <- function(probs.l,
                                      y.. = NULL,
                                      idces = NULL,
                                      scale.rowsum.to.1 = TRUE){

  probs <- do.call(rbind, probs.l)

  if(!is.null(idces)){

    probs <- probs[match(1:nrow(probs), idces),]

  }

  #probs <- probs[, levels(y..)]

  if(scale.rowsum.to.1 == TRUE){
    probs.rowsum1 <- t(apply(probs, 1, function(x) x/sum(x)))
  }else{
    probs.rowsum1 <-  probs
  }

  y.p.rowsum1 <- colnames(probs.rowsum1)[apply(probs.rowsum1,1, which.max)]
  y.l <- list(y.p.rowsum1)

  #Misclassification Error
  err.misc.l <- lapply(y.l, subfunc_misclassification_rate, y.true.class = y..)
  err.misc <- unlist(err.misc.l)

  #AUC HandTIll2001
  results.sc.p.rowsum1.l <- list(probs.rowsum1)
  auc.HT2001.l <- lapply(results.sc.p.rowsum1.l, subfunc_multiclass_AUC_HandTill2001, y.true.class = y..)
  auc.HT2001 <- unlist(auc.HT2001.l)

  #Brier
  brierp.rowsum1 <- brier(scores = probs.rowsum1, y = y..)

  #mlogloss
  loglp.rowsum1 <- mlogloss(scores = probs.rowsum1, y = y..)

  # Results
  res <- list(misc.error = err.misc,
              auc.HandTill = auc.HT2001,
              brier = brierp.rowsum1,
              mlogloss = loglp.rowsum1)
  return(res)
}




#'Calibration for the cross validation results of the raw model
#'
#'Evaluate the cross validation performance of various machine learning models
#'after calibration.
#'
#'@param out.path For all the calibration cross validation loops, their result
#'  files will be saved in the folder set by this parameter. It is the folder
#'  name will be created in the current working directory.
#'@param out.fname The final cross validation result files will be saved in
#'  the folder set by the parameter \code{out.path} and the name prefix of the
#'  result files need to be set by this parameter, default is "probsCVfold".
#'@param y.. The true labels of the samples. Can be a vector, factor, or NULL.
#'  If it is a vector or factor, each element is a label for a sample and its
#'  element order should be the same as the sample order in the data used for
#'  raw model cross validation performed by the function \code{maincv}. If it
#'  is NULL, there should be a vector or factor named `y` in the environment
#'  and the function will load it as the sample true labels.
#'@param load.path The calibration model cross validation performed by this
#'  function needs the raw model cross validation results generated by the
#'  function \code{maincv}, and this parameter is used to transfer the path
#'  with the raw model cross validation results. It can be an absolute path
#'  or a relative one.
#'@param load.fname The name prefix of the raw cross validation result files
#'  generated by the function \code{maincv}.
#'@param algorithm The algorithm of the raw model need to be calibrated here.
#'  Should be one of "RF", "SVM", "XGB", "eSVM", "eNeural", and "MOGONET". The
#'  default value is "eSVM".
#'@param normalcv Indicating whether the calibration cross validation loops
#'  here should be normal or nested. If the raw model cross validation loops
#'  are nested, this parameter can be set as either TRUE or FALSE. For FALSE,
#'  the calibration loops here will be the same as the raw model loops. For
#'  TRUE, only the outer loops of the raw model cross validation will be used
#'  here as the normal calibration loops. If the raw model loops are normal,
#'  this parameter can only be TRUE. Default is FALSE.
#'@param brglm.ctrl.max.iter A parameter special for the Firth's penalized
#'  calibration. It defines the maximum iteration number for fitting the bias
#'  reduced logistic regression model.
#'@param cores Number of threads used for parallization. It only accelerates
#'  the ridge calibration step. Default is 1.
#'@param setseed The random seed for the ridge calibration because it needs to
#'  do a grid search for the hyperparameter lambda, and this will be conducted
#'  via a hyperparameter search cross validation process with its random seed
#'  set by this parameter. Default is 1234.
#'@return Will return a matrix recording the cross validation performance of
#'  different raw model and calibration model combinations, including that of
#'  the raw model, the raw + logistic regrssion, the raw + Firth's regression,
#'  and the raw + ridge model. In addition, the calibration score matrixes of
#'  the cross validation loops will be saved in the path set by the parameter
#'  \code{out.path}.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'maincv(y.. = labels, betas.. = betas, subset.CpGs = 10000, n.cv.folds = 5, 
#'  normalcv = TRUE, out.path = 'RFCV', out.fname = 'CVfold', 
#'  method = 'RF', seed = 1234, cores = 4)
#'
#'RFres <- maincalibration(y.. = labels, load.path = 'RFCV', 
#'  load.fname = 'CVfold', normalcv = TRUE, algorithm = 'RF', 
#'  setseed = 1234)
#'  
#'\dontrun{
#'maincv(y.. = labels, betas.. = betas, subset.CpGs = 10000, n.cv.folds = 5, 
#'  normalcv = TRUE, out.path = 'eSVMCV', out.fname = 'CVfold', 
#'  method = 'eSVM', seed = 1234, cores = 10)
#'  
#'eSVMres <- maincalibration(y.. = labels, load.path = 'eSVMCV', 
#'  load.fname = 'CVfold', normalcv = TRUE, algorithm = 'eSVM', 
#'  setseed = 1234)
#'}
#'@export
maincalibration <- function(out.path = "calibration/",
                            out.fname = "probsCVfold",

                            y.. = NULL,

                            load.path,
                            load.fname,

                            algorithm = "eSVM",

                            normalcv = FALSE,

                            brglm.ctrl.max.iter = 10000,

                            cores = 1,
                            setseed = 1234){

  load.path.w.name <- file.path(load.path, load.fname)
  if(algorithm == 'RF'){
    algorithm <- 'rf'
  }else if(algorithm == 'SVM'){
    algorithm <- 'svm'
  }else if(algorithm == 'XGB'){
    algorithm <- 'xgboost'
  }else if(algorithm == 'eSVM'){
    algorithm <- 'esvm'
  }else if(algorithm == 'eNeural'){
    algorithm <- 'eneural'
  }else if(algorithm == 'MOGONET'){
    algorithm <- 'mogonet'
  }
  saveres <- TRUE
  scale.rowsum.to.1 <- TRUE

  cvfiles <- dir(load.path)
  cvfiles <- cvfiles[grepl(pattern=paste0(load.fname, '.*\\.RData'), x = cvfiles)]

  ks <- gsub(pattern = '\\.RData', replacement = '', x = cvfiles)
  ks <- gsub(pattern = '^.*\\.', replacement = '', x = ks)
  kmax <- max(as.numeric(ks))

  Ks <- gsub(pattern = '\\.RData', replacement = '', x = cvfiles)
  Ks <- unlist(lapply(X = strsplit(x = Ks, split = '\\.', fixed = FALSE),
                      FUN = function(x){x[length(x) - 1]}))
  Kmax <- max(as.numeric(Ks))

  if(kmax == 0){
    normalcv <- TRUE
  }



  if(is.null(y..) & exists("y")){
    y.. <- get("y", envir = .GlobalEnv)
  }else if(is.null(y..) & !exists("y")){
    stop("Please provide the true class labels vector (y) for further analyses\n")
  }

  if(!is.factor(y..)){
    y.. <- as.character(y..)
    freqy <- table(y..)
    freqy <- freqy[order(-freqy)]
    y.. <- factor(y.., levels = names(freqy), ordered = TRUE)

  }


  scores.l <- list()
  idces <- c()

  if(sum(c('rf', 'svm', 'xgboost', 'esvm', 'eneural', 'mogonet') %in% algorithm) > 0){

    probs.lr.l <- list()
    probs.flr.l <- list()
    probs.mr.l <- list()

  }

  i <- 1
  for(i in 1:Kmax){
    # Create loading environment for safe loading the RData files of outerfolds
    env2load.outer <- environment()

    load(paste0(load.path.w.name, ".", i, ".", 0, ".RData"), envir = env2load.outer)

    outerfold <- fold

    if(algorithm == "rf"){
      scores <- get("scores", envir = env2load.outer)
    }else if(algorithm == "svm"){
      scores <- get("scores.pred.svm.e1071.mtx", envir = env2load.outer)
    }else if(algorithm == "xgboost"){
      scores <- get("scores.pred.xgboost", envir = env2load.outer)
    }else if(algorithm == "esvm"){
      scores <- get("scores.pred.eSVM", envir = env2load.outer)
    }else if(algorithm == "eneural"){
      scores <- get("scores.pred.eNeural", envir = env2load.outer)
    }else if(algorithm == 'mogonet'){
      scores <- get('scores', envir = env2load.outer)
    }else{
      scores <- get('probs', envir = env2load.outer)
    }

    # Re-assign to new variable
    scores.i.0.outertest <- scores
    idx.i.0.outertest <- env2load.outer$fold$test
    y.i.0.outertest <- y..[idx.i.0.outertest]

    if(sum(c('rf', 'svm', 'xgboost', 'esvm', 'eneural', 'mogonet') %in% algorithm) > 0){

      if(normalcv == FALSE){

        scoresl <- list()
        idxl <- list()
        j <- 1
        for(j in 1:kmax){
          # Create loading environment for safe loading the RData files of nested inner folds
          env2load.inner <- environment()

          load(paste0(load.path.w.name, ".", i, ".", j, ".RData"))

          if(algorithm == "rf"){scores <- get("scores", envir = env2load.inner)}
          if(algorithm == "svm"){scores <- get("scores.pred.svm.e1071.mtx", envir = env2load.inner)}
          if(algorithm == "xgboost"){scores <- get("scores.pred.xgboost", envir = env2load.inner)}
          if(algorithm == "esvm"){scores <- get("scores.pred.eSVM", envir = env2load.inner)}
          if(algorithm == "eneural"){scores <- get("scores.pred.eNeural", envir = env2load.inner)}
          if(algorithm == 'mogonet'){scores <- get('scores', envir = env2load.inner)}

          scoresl[[j]] <- scores
          idxl[[j]] <- env2load.inner$fold$test

        }

        #Collapse lists in to matrix objects

        scores.i.j.innertest.all <- do.call(rbind, scoresl)

        #scores.i.j.innertest.all <- do.call(rbind, scoresl)

        idx.i.j.innertest.all <- unlist(idxl)
        y.i.j.innertest.all <- y..[idx.i.j.innertest.all]


        message("Training calibration model using Platt scaling by LR @ ", Sys.time(), "\n")

        probs.platt.diagn.l <- list()
        c <- 1

        for(c in seq_along(colnames(scores.i.j.innertest.all))){

          diagn <- colnames(scores.i.j.innertest.all)[c]

          platt.calfit <- subfunc_Platt_train_calibration(y.i.j.innertest = y.i.j.innertest.all,
                                                          scores.i.j.innertest = scores.i.j.innertest.all,
                                                          diagn.class = diagn)


          probs.platt.diagn.l[[c]] <- subfunc_Platt_fit_testset(scores.i.0.outertest = scores.i.0.outertest,
                                                                calib.model.Platt.diagn.i = platt.calfit,
                                                                diagn.class = diagn)

        }

        probs.lr <- do.call(cbind, probs.platt.diagn.l)

        colnames(probs.lr) <- colnames(scores)
        #colnames(probs.lr) <- levels(y.i.0.outertest)
        probs.lr <- probs.lr[, levels(y.i.0.outertest), drop = FALSE]


        message("Training calibration model using Firth's penalized LR @ ", Sys.time(), "\n")

        probs.platt.firth.brglm.l <- list()

        c <- 1

        for(c in seq_along(colnames(scores.i.j.innertest.all))){

          diagn <- colnames(scores.i.j.innertest.all)[c]

          platt.brglm.calfit <- subfunc_Platt_train_calibration_Firth(y.i.j.innertest = y.i.j.innertest.all,
                                                                      scores.i.j.innertest = scores.i.j.innertest.all,
                                                                      diagn.class = diagn,
                                                                      brglm.control.max.iteration = brglm.ctrl.max.iter)

          probs.platt.firth.brglm.l[[c]] <- subfunc_Platt_fit_testset_Firth(scores.i.0.outertest = scores.i.0.outertest,
                                                                            calib.model.Platt.Firth.diagn.i = platt.brglm.calfit,
                                                                            diagn.class = diagn)


        }

        probs.flr <- do.call(cbind, probs.platt.firth.brglm.l)

        colnames(probs.flr) <- colnames(scores)
        probs.flr <- probs.flr[, levels(y.i.0.outertest), drop = FALSE]


        message("Training the multinomial ridge (MR) calbriation model @ ", Sys.time(), "\n")

        set.seed(setseed)

        if(cores > 1){

          parallel.cv.glmnet <- TRUE

          threads <- parallel::detectCores()
          cl <- parallel::makeCluster(min(cores, threads))

          doParallel::registerDoParallel(cl)

        }else{

          parallel.cv.glmnet <- FALSE

        }

        suppressWarnings(
          cv.calfit <- glmnet::cv.glmnet(y = y.i.j.innertest.all,
                                         x = scores.i.j.innertest.all,
                                         family = "multinomial",
                                         type.measure = "mse",
                                         alpha = 0,
                                         nlambda = 100,
                                         lambda.min.ratio = 10^-6,
                                         parallel = parallel.cv.glmnet)
        )

        if(cores > 1){

          parallel::stopCluster(cl)

          unregister_dopar()

        }

        probs.mr <- predict(cv.calfit$glmnet.fit,
                            newx = scores.i.0.outertest,
                            type = "response",
                            s = cv.calfit$lambda.1se)[,,1]

        probs.mr <- probs.mr[, levels(y.i.0.outertest), drop = FALSE]




      }else{

        scoresl <- list()
        idxl <- list()

        iseqs <- 1:Kmax
        otheriseqs <- setdiff(iseqs, i)
        j <- 1
        for(j in otheriseqs){

          env2load.other <- environment()

          message("\nLoad the test set from the ", j, ".0 outer fold @ ", Sys.time(), "\n")

          load(paste0(load.path.w.name, ".", j, ".", 0, ".RData"), envir = env2load.other)

          if(algorithm == "rf"){scores <- get("scores", envir = env2load.other)}
          if(algorithm == "svm"){scores <- get("scores.pred.svm.e1071.mtx", envir = env2load.other)}
          if(algorithm == "xgboost"){scores <- get("scores.pred.xgboost", envir = env2load.other)}
          if(algorithm == "esvm"){scores <- get("scores.pred.eSVM", envir = env2load.other)}
          if(algorithm == "eneural"){scores <- get("scores.pred.eNeural", envir = env2load.other)}
          if(algorithm == 'mogonet'){scores <- get('scores', envir = env2load.other)}

          scoresl[[j]] <- scores
          idxl[[j]] <- env2load.other$fold$test

        }

        # Collapse lists in to matrix objects
        scores.j.0.outertest.all <- do.call(rbind, scoresl)
        idx.j.0.outertest.all <- unlist(idxl)
        y.j.0.outertest.all <- y..[idx.j.0.outertest.all]


        message("Training calibration model using Platt scaling by LR @ ", Sys.time(), "\n")

        # Platt scaling with LR
        probs.platt.diagn.l <- list()
        c <- 1

        for(c in seq_along(colnames(scores.j.0.outertest.all))){

          diagn <- colnames(scores.j.0.outertest.all)[c]

          platt.calfit <- subfunc_Platt_train_calibration(y.i.j.innertest = y.j.0.outertest.all,
                                                          scores.i.j.innertest = scores.j.0.outertest.all,
                                                          diagn.class = diagn)

          probs.platt.diagn.l[[c]] <- subfunc_Platt_fit_testset(scores.i.0.outertest = scores.i.0.outertest,
                                                                calib.model.Platt.diagn.i = platt.calfit,
                                                                diagn.class = diagn)

        }

        probs.lr <- do.call(cbind, probs.platt.diagn.l)

        colnames(probs.lr) <- colnames(scores)
        #colnames(probs.lr) <- levels(y.i.0.outertest)
        probs.lr <- probs.lr[, levels(y.i.0.outertest), drop = FALSE]

        message("Training calibration model using Firth's penalized LR @ ", Sys.time(), "\n")

        # Platt scaling with Firth's penalized LR (FLR)
        probs.platt.firth.brglm.l <- list()
        c <- 1

        for(c in seq_along(colnames(scores.j.0.outertest.all))){

          diagn <- colnames(scores.j.0.outertest.all)[c]

          platt.brglm.calfit <- subfunc_Platt_train_calibration_Firth(y.i.j.innertest = y.j.0.outertest.all,
                                                                      scores.i.j.innertest = scores.j.0.outertest.all,
                                                                      diagn.class = diagn,
                                                                      brglm.control.max.iteration = brglm.ctrl.max.iter)

          probs.platt.firth.brglm.l[[c]] <- subfunc_Platt_fit_testset_Firth(scores.i.0.outertest = scores.i.0.outertest,
                                                                            calib.model.Platt.Firth.diagn.i = platt.brglm.calfit,
                                                                            diagn.class = diagn)


        }

        probs.flr <- do.call(cbind, probs.platt.firth.brglm.l)

        colnames(probs.flr) <- colnames(scores)
        probs.flr <- probs.flr[, levels(y.i.0.outertest), drop = FALSE]


        message("Training the multinomial ridge (MR) calbriation model @ ", Sys.time(), "\n")
        # Fit multinomial ridge regression (MR) model
        set.seed(setseed)

        if(cores > 1){

          parallel.cv.glmnet <- TRUE

          threads <- parallel::detectCores()
          cl <- parallel::makeCluster(min(cores, threads))

          doParallel::registerDoParallel(cl)


        }else{

          parallel.cv.glmnet <- FALSE

        }

        suppressWarnings(
          cv.calfit <- glmnet::cv.glmnet(y = y.j.0.outertest.all,
                                         x = scores.j.0.outertest.all,
                                         family = "multinomial",
                                         type.measure = "mse",
                                         alpha = 0,
                                         nlambda = 100,
                                         lambda.min.ratio = 10^-6,
                                         parallel = parallel.cv.glmnet)
        )

        if(cores > 1){

          parallel::stopCluster(cl)

          unregister_dopar()

        }

        probs.mr <- predict(cv.calfit$glmnet.fit,
                            newx = scores.i.0.outertest,
                            type = "response",
                            s = cv.calfit$lambda.1se)[,,1]

        probs.mr <- probs.mr[, levels(y.i.0.outertest), drop = FALSE]

      }

    }

    idx <- outerfold$test
    idces <- c(idces, idx)

    errs <- sum(colnames(scores.i.0.outertest)[apply(scores.i.0.outertest, 1, which.max)] != y..[outerfold$test])/length(outerfold$test)
    message("Misclassification error (", algorithm, ") raw scores on the ", i, ".0 fold: ", errs, "\n")


    folder.path <- file.path(getwd(), out.path)
    dir.create(folder.path, recursive = TRUE)

    scores <- scores.i.0.outertest

    if(saveres == TRUE){
      save(scores, errs,
           file = file.path(folder.path, paste(out.fname, algorithm, "raw", i, 0, "RData", sep = ".")))
    }

    if(sum(c('rf', 'svm', 'xgboost', 'esvm', 'eneural', 'mogonet') %in% algorithm) > 0){

      errp <- sum(colnames(probs.lr)[apply(probs.lr, 1, which.max)] != y..[outerfold$test])/length(outerfold$test)
      message("Misclassification error of Platt-LR-calibrated (", algorithm, ") probabilities on the ", i, ".0 fold: ", errp, "\n")


      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = TRUE)

      if(saveres == TRUE){
        save(probs.lr, scores, errs, errp,
             file = file.path(folder.path, paste(out.fname, algorithm, "LR", i, 0, "RData", sep = ".")))
      }


      errp <- sum(colnames(probs.flr)[apply(probs.flr, 1, which.max)] != y..[outerfold$test])/length(outerfold$test)
      message("Misclassification error of Platt-FLR-calibrated (", algorithm, ") probabilities on the ", i, ".0 fold: ", errp, "\n")

      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = TRUE)

      if(saveres == TRUE){
        save(probs.flr, scores, errs, errp,
             file = file.path(folder.path, paste(out.fname, algorithm, "FLR", i, 0, "RData", sep = ".")))
      }


      errp <- sum(colnames(probs.mr)[apply(probs.mr, 1, which.max)] != y..[outerfold$test])/length(outerfold$test)
      message("Misclassification error of MR-calibrated (", algorithm, ") probabilities on the ", i, ".0 fold: ", errp, "\n")

      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = TRUE)

      if(saveres == TRUE){
        save(probs.mr, scores, errs, errp,
             file = file.path(folder.path, paste(out.fname, algorithm, "MR", i, 0, "RData", sep = ".")))
      }

      probs.lr.l[[i]] <- probs.lr
      probs.flr.l[[i]] <- probs.flr
      probs.mr.l[[i]] <- probs.mr


    }


    scores.l[[i]] <- scores


  }

  message("Start performance evaluation @ ", Sys.time(), "\n")

  rawres <- sub_performance_evaluator(probs.l = scores.l,
                                      y.. = y..,
                                      idces = idces,
                                      scale.rowsum.to.1 = scale.rowsum.to.1)

  message("Misclassification Error (", algorithm, ") raw scores: ", rawres$misc.error, "\n")
  message("Multiclass AUC (Hand&Till 2001) (", algorithm, ") raw scores: ", rawres$auc.HandTill, "\n")
  message("Brier score (BS) (", algorithm, ") raw scores: ", rawres$brier, "\n")
  message("Multiclass log loss (LL) (", algorithm, ") raw scores: ", rawres$mlogloss, "\n")

  res <- do.call(rbind, list(rawres))

  row.names(res) <- paste0(algorithm, c(''))

  if(sum(c('rf', 'svm', 'xgboost', 'esvm', 'eneural', 'mogonet') %in% algorithm) > 0){

    lrres <- sub_performance_evaluator(probs.l = probs.lr.l,
                                       y.. = y..,
                                       idces = idces,
                                       scale.rowsum.to.1 = scale.rowsum.to.1)

    message("Misclassification Error of Platt-LR-calibrated (", algorithm, ") probabilities: ",
            lrres$misc.error, "\n")
    message("Multiclass AUC (Hand&Till 2001) of Platt-LR-calibrated (", algorithm, ") probabilities: ",
            lrres$auc.HandTill, "\n")
    message("Brier score (BS) of Platt-LR-calibrated (", algorithm, ") probabilities: ",
            lrres$brier, "\n")
    message("Multiclass log loss (LL) of Platt-LR-calibrated (", algorithm, ") probabilities: ",
            lrres$mlogloss, "\n")

    flrres <- sub_performance_evaluator(probs.l = probs.flr.l,
                                        y.. = y..,
                                        idces = idces,
                                        scale.rowsum.to.1 = scale.rowsum.to.1)

    message("Misclassification Error of Platt-FLR-calibrated (", algorithm, ") probabilities: ",
            flrres$misc.error, "\n")
    message("Multiclass AUC (Hand&Till 2001) of Platt-FLR-calibrated (", algorithm, ") probabilities: ",
            flrres$auc.HandTill, "\n")
    message("Brier score (BS) of Platt-FLR-calibrated (", algorithm, ") probabilities: ",
            flrres$brier, "\n")
    message("Multiclass log loss (LL) of Platt-FLR-calibrated (", algorithm, ") probabilities: ",
            flrres$mlogloss, "\n")

    mrres <- sub_performance_evaluator(probs.l = probs.mr.l,
                                       y.. = y..,
                                       idces = idces,
                                       scale.rowsum.to.1 = scale.rowsum.to.1)

    message("Misclassification Error of MR-calibrated (", algorithm, ") probabilities: ",
            mrres$misc.error, "\n")
    message("Multiclass AUC (Hand&Till 2001) of MR-calibrated (", algorithm, ") probabilities: ",
            mrres$auc.HandTill, "\n")
    message("Brier score (BS) of MR-calibrated (", algorithm, ") probabilities: ",
            mrres$brier, "\n")
    message("Multiclass log loss (LL) of MR-calibrated (", algorithm, ") probabilities: ",
            mrres$mlogloss, "\n")


    res <- do.call(rbind, list(rawres,
                               lrres,
                               flrres,
                               mrres))

    row.names(res) <- paste0(algorithm, c('', '_LR', '_FLR', '_MR'))


  }

  message("Finished @ ", Sys.time())

  return(res)

}


#Train


#'Train various classification models
#'
#'Train classification models without or with various calibraion methods
#'
#'@param y.. The true labels of the samples. Can be a vector, factor, or NULL.
#'  If it is a vector or factor, each element is a label for a sample and the
#'  element order in it should be the same as the sample order in the sample
#'  data provided by the parameter \code{betas..}. If it is NULL, there should
#'  be a vector or factor named `y` in the global environment and the function
#'  will load it and use it as the sample true labels.
#'@param betas.. The beta value matrix of the samples. Each row is one sample
#'  and each column is one feature. It can also be set as NULL, and in this
#'  case, the function will read the data via the directory \code{betas.path}.
#'  The absolute path of the file of these data should be provided by the
#'  parameter \code{betas.path}.
#'@param betas.path If the parameter \code{betas..} is NULL, this parameter is
#'  necessary to provide the file path of the betas matrix data, so that the
#'  function will read the data from this path. It should be an absolute path
#'  string, and the file should be an .rds file.
#'@param subset.CpGs The feature selection method. It can be a numeric number
#'  such as 10000, and then the top 10000 most variable features of the data
#'  will be selected as the features to construct the machine learning model.
#'  It can also be the string "limma", so that \code{limma} will be used to
#'  select the significantly differential features between a sample class and
#'  all other samples. The differential ones should fulfill the condition that
#'  their adjusted p-value < \code{padjcut} (default is 0.05), and the betas
#'  value difference between the sample class and other samples should be > 
#'  \code{xcutoff}, or < -\code{xcutoff} (default value is 0.1). After the 
#'  differential features for each class have been found by \code{limma}, all
#'  of them will be mixed together and ordered by their adjusted p-value and
#'  the absolute of beta value difference, and then the top \code{cutnum} ones
#'  (default number is 10000) will be selected and as the final features. In
#'  addition, if there are any confounding factors in the dataset need to be
#'  removed, they can be provided by the parameter \code{confoundings} and
#'  \code{limma} will select the features after these confoundings have been
#'  adjusted. The parameter \code{confoundings} should be provided as a vector
#'  with the confounding names in the meta data (the meta data are provided by
#'  the parameter \code{anno}) as elements. As to \code{subset.CpGs}, it can
#'  also be the string "SCMER", and then \code{SCMER} will be used to select
#'  features from the data. In this case, other parameters should be set well,
#'  including \code{lasso}, \code{ridge}, \code{n_pcs}, \code{perplexity},
#'  \code{savefigures}, \code{pythonpath} and \code{topfeaturenumber}. Because
#'  \code{SCMER} is a method to select features able to preserve the original
#'  manifold structure of the data after its feature selection by elastic net,
#'  most of these parameters are used to config the manifold (\code{n_pcs}
#'  and \code{perplexity}) and elastic net (\code{lasso}, and \code{ridge}),
#'  while another important one is \code{pythonpath} that is used to tell the
#'  function the absolute path of \code{Python} you want to run \code{SCMER}
#'  because this method also depends on \code{Python}. In addition, because
#'  \code{limma} and \code{SCMER} can be time-consuming if running on a large
#'  number of candidate features, it is recommended to do a prescreen before
#'  running them, and the parameter \code{topfeaturenumber} can be set as a
#'  numeric value such as 50000, so that the top 50000 most variable features
#'  will be selected, and the top variable, \code{limma}, or the \code{SCMER}
#'  features can be selected further on the prescreened data. The parameter
#'  \code{subset.CpGs} can also be set as NULL, so that feature selection will
#'  not be performed before the model construction.
#'@param seed Some process performed by this function will need a seed number
#'  to fix the random process, such as the random sampling steps of some models
#'  such as random forest, eSVM, eNeural, etc, and this parameter is used to
#'  set their random seeds. Default value is 1234.
#'@param cores The core number need to do parallelization computation. Default
#'  is 10.
#'@param topfeaturenumber As mentioned in the \code{subset.CpGs} parameter
#'  part, it is used to set the prescreened feature number. Default is 50000.
#'  It can also be set as NULL, so that no precreen will be done on the data.
#'@param savefeaturenames Default is FALSE, but if is set as TRUE, the feature
#'  names selected by the feature selection process will be saved as a vector.
#'@param method Which algorithm need to be used to train the model. Can be a
#'  string as "RF", "SVM", "XGB", "ENet", "eNeural", "MOGONET", or "eSVM". The
#'  default value is "eSVM".
#'@param anno A data frame recording the meta data of the samples, and should
#'  contain at least 2 columns named as "label" and "sentrix". The former one
#'  records the sample labels while the latter one records the sample IDs that
#'  also used as row names of the methylation data matrix. The default value
#'  is NULL and it is not necessary as long as the \code{y..} parameter is
#'  provided, but if need to use \code{limma} to do the feature selection and
#'  remove the confounding factors, it should be provided with the confounding
#'  factors included in it.
#'@param lasso A parameter special for \code{SCMER} feature selection and it
#'  defines the strength of L1 regularization in the elastic net process of
#'  \code{SCMER}. Default is 3.25e10-7, so that around 10000 features will be
#'  selected from 50000 prescreened candidate features.
#'@param ridge A parameter special for \code{SCMER} feature selection and it
#'  defines the strength of L2 regularization in the elastic net process of
#'  \code{SCMER}. Default is 0, so that the elastic net process is actually
#'  a LASSO process.
#'@param n_pcs Number of principle components need to reconstruct the sample-
#'  sample distance matrix during the \code{SCMER} selection. Default is 100.
#'@param perplexity Perplexity of tSNE modeling for the \code{SCMER} feature
#'  selection. Default is 10.
#'@param savefigures Whether save the PCA and UMAP figures generated by the
#'  \code{SCMER} method or not. Choose from TRUE and FALSE. Default is FALSE.
#'@param pythonpath Because the feature selection method \code{SCMER} and the
#'  model training algorithm \code{MOGONET} are \code{Python} based methods,
#'  the directory of the \code{Python} interpreter you want to use to run them
#'  should be provided via this parameter, and to run \code{SCMER}, several
#'  modules should be installed to the \code{Python} environment, including
#'  \code{time}, \code{functiontools}, \code{abc}, \code{torch}, \code{numpy},
#'  \code{typing}, \code{pandas}, \code{matplotlib}, \code{multiprocessing},
#'  \code{scanpy}, and \code{sklearn}. To run \code{MOGONET}, the modules are
#'  \code{numpy}, \code{sklearn}, and \code{torch}.
#'@param confoundings A parameter special for \code{limma} feature selection.
#'  Details can be seen in the \code{subset.CpGs} parameter section.
#'@param padjcut A parameter for \code{limma} feature selection. Default value
#'  is 0.05 and details can be seen in the \code{subset.CpGs} section.
#'@param xcutoff A parameter for \code{limma}. Details can also be seen in the
#'  \code{subset.CpGs} section. Default value is 0.1.
#'@param cutnum A parameter special for \code{limma}. Details can be seen in
#'  the \code{subset.CpGs} section.
#'@param ntrees A parameter special for the random forest (RF) model, defining
#'  the number of decision trees in the RF model. Default is 500.
#'@param p A parameter special for RF. In the RF method here, a 2-step process
#'  is conducted. The first one constructs an RF model on all the candidate
#'  features with \code{ntrees} trees. Then, the top \code{p} most important
#'  features in this step will be selected by calculating their influence on
#'  the error of each tree using permuted out-of-bag data, and these features
#'  will be transferred to the second step to construct a second RF model on
#'  them, also with a tree number of \code{ntrees}. The parameter \code{p} is
#'  used to control how many top important features are needed to be selected,
#'  and the default value is 200.
#'@param modelcv For the models of SVM, eSVM, XGBoosting (XGB), elastic net
#'  (ENet) and eNeural, a hyperparameter search step is performed to find the
#'  optimal hyperparameters, via cross validation, such as the regularization
#'  constant of SVM, eSVM, and ENet, and this parameter is used to define the
#'  number of cross validation loops for hyperparameter search. Default is 5,
#'  and it means to train a model, the data will be divided into 5 sets and a
#'  5-fold cross validation will be used to evaluate the performance of the
#'  models with different hyperparameters and finally chooses the optimal one.
#'  Hence, before the model training, a 5 fold cross validation will be done
#'  first to find the optimal hyperparameter.
#'@param C.base A parameter special for SVM and eSVM to set the regularization
#'  constant. This constant will be calculated by the function as base^index,
#'  and \code{C.base} here serves as the base number. Combined with other 2
#'  parameters \code{C.min} and \code{C.max} serving as indexes, it defines a
#'  regularization constant series. Its start is \code{C.base}^\code{C.min},
#'  and the end is \code{C.base}^\code{C.max}, while the near elements of the
#'  series have a difference of \code{C.base} fold. If the 2 indexes are set
#'  as the same, the series will become 1 regularization constant. The default
#'  value of \code{C.base} is 10.
#'@param C.min As mentioned in the \code{C.base} part, this parameter is used
#'  as the index of the small regularization constant number to set a series
#'  for SVM and eSVM. Default is -3.
#'@param C.max As mentioned in the \code{C.base} part, this parameter is used
#'  as the index of the large regularization constant number to set a series
#'  for SVM and eSVM. Default is -2.
#'@param learnernum A parameter special for eSVM, eNeural and \code{MOGONET}
#'  to set their base learner number. Default is 10.
#'@param minlearnersize A parameter special for eSVM, eNeural, \code{MOGONET}
#'  to define the lower limit of the feature number of their base learners.
#'  Default value is 1000, meaning each base learner should have at least 1000
#'  features after the random sampling process to sample features for them.
#'@param viewstandard When this parameter is set as NULL. The features will be
#'  assigned to the base learners of eSVM, eNeural and \code{MOGONET} through
#'  random sampling. While if it is "Relation_to_Island" and the features are
#'  DNA methylation probes, they will be split into groups of island probes,
#'  N shelf and N shore probes, S shelf and S shore probes, and opensea probes
#'  and then for each base learner, its features will be sampled from one of
#'  these groups. If this parameter is set as "UCSC_RefGene_Group", then the
#'  probes will be grouped into promoter probes, gene body probes and other
#'  probes and each base learner will get its features via sampling on one of
#'  these groups. The default value of this parameter is NULL.
#'@param platform When \code{viewstandard} is set as "Relation_to_Island" or
#'  "UCSC_RefGene_Group", this parameter will be used to define the platform
#'  of the probe annotation information to split them into different groups.
#'  The default value is "450K", and can also be "EPIC".
#'@param max_depth A parameter special for XGB. Its the maximum depth of each
#'  tree. Default is 6.
#'@param eta A parameter special for XGB. It controls the learning rate via
#'  scaling the contribution of each tree by a factor of 0 < eta < 1 when the
#'  tree is added to the approximation, and can prevent overfitting by making
#'  the boosting process more conservative. Its default value is a vector of
#'  \code{c(0.1, 0.3)}, meaning a grid search will be conducted between these
#'  2 values to find the optimal \code{eta} value with less misclassification
#'  rate.
#'@param gamma A parameter special for XGB. Defines the minimum loss reduction
#'  required to make a further partition on a leaf node of the tree. Default
#'  value is a vector of \code{c(0, 0.01)} and a grid search will be conducted
#'  on it.
#'@param colsample_bytree Special for XGB. Defines subsample ratio of columns
#'  when constructing each tree. Default is \code{c(0.01, 0.02, 0.05, 0.2)},
#'  and a grid search will be performed on it.
#'@param subsample Subsample ratio of the training instance for XGB. Setting
#'  it to 0.5 means that XGB randomly collected half of the data instances to
#'  grow trees and this can prevent overfitting and make computation shorter.
#'  Its default value is 1.
#'@param min.chwght Minimum sum of instance weight (hessian) needed in a child
#'  of the tree in XGB. If the tree partition step results in a leaf node with
#'  the sum of instance weight less than this value, the building process will
#'  give up further partitioning. Default is 1.
#'@param nrounds A parameter special for XGB. Defines the max number of the
#'  boosting iterations. Default is 100.
#'@param early_stopping_rounds Special for XGB and default value is 50, which
#'  means the training with a validation set will stop if the performance dose
#'  not improve for 50 rounds.
#'@param alpha.min. A parameter special for ENet. Need to use with the other 2
#'  parameters \code{alpha.max.} and \code{by.} to set an elastic net mixing
#'  parameter series. The default value of \code{alpha.min.} is 0, the default
#'  value of \code{alpha.max.} is 1, and the default value of \code{by.} is
#'  0.1, so that a mixing parameter series staring with 0, ending with 1, and
#'  with a difference between its neighbor elements as 0.1 will be generated
#'  and to do a grid search on it to select the optimal mixing parameter value
#'  (alpha) giving the smallest MSE across the hyperparameter searching cross
#'  validation and then it will be used for the next model training.
#'@param alpha.max. A parameter special for ENet. Need to use with the other 2
#'  parameters \code{alpha.min.} and \code{by.} to set an elastic net mixing
#'  parameter series. As mentioned in the \code{alpha.min.} section.
#'@param by. A parameter special for ENet. Detail is in the \code{alpha.min.}
#'  section.
#'@param predefinedhidden A parameter special for eNeural. Use it to transfer
#'  the node number of each hidden layer of one neural network in the eNeural
#'  model. Need to be a vector, such as \code{c(100, 50)}, so that for each
#'  neural network in the eNeural model, 2 hidden layers will be set up. One
#'  is with 100 nodes, while the other is with 50 ones. Default value is NULL,
#'  so that the function will set up a hidden layer structure automatically.
#'  If the parameter \code{gridsearch} is set as FALSE, this structure is with
#'  2 layers and the node number of them are both around 1/100 of the input
#'  node number. If \code{gridsearch} is TRUE, several different hidden layer
#'  structures will be generated and a search will be performed on them to get
#'  the optimal one.
#'@param maxepchs A parameter special for eNeural and defines the epoch number
#'  for the neural network training. If the parameter \code{gridsearch} is set
#'  as FALSE, the epoch number is fixed as this, but if \code{gridsearch} is
#'  TRUE, an epoch number series will be set up starting from 10 and ending at
#'  \code{maxepchs}, with the neighbor elements having a 10 fold difference.
#'  Then, grid search will be performed across this series. The default value
#'  of \code{maxepchs} is 10.
#'@param activation Activation function special for eNeural. Can be a string
#'  or a vector of strings to do grid search. Default is "Rectifier". Can also
#'  be "Tanh" or "Maxout", or a vector with elements from them.
#'@param momentum_start Special for eNeural. Defines the initial momentum at
#'  the beginning of training (try 0.5). Default is 0. And a vector covering
#'  different values can be used for hyperparameter grid search.
#'@param rho Speical parameter for eNeural. Adaptive learning rate time decay
#'  factor (defines the similarity to prior updates). Default is 0.99. Can be
#'  a vector for hyperparameter grid search.
#'@param gridsearch Special parameter for eNeural. Whether do grid search to
#'  select the optimal hyperparameters, or directly use the fixed and given
#'  hyperparameters to train the neural networks. If it is TRUE, grid search
#'  will be performed on the hyperparameters of hidden layer size and depth,
#'  epoch number, activation function, initial momentum, and adaptive learning
#'  rate time decay factor.
#'@param mogonettestbetas Special parameter for \code{MOGONET}. The validation
#'  data should be provided, as rows as samples and columns as features. The
#'  uniqueness of \code{MOGONET} from other algorithms here is that it is a
#'  graph-based method, so the validation data with unknown labels should be
#'  trained together with the training data and after that their labels can be
#'  predicted. This is different from other algorithms because the validation
#'  data don't need to be provided during their model training stage and the
#'  labels can be predicted using the trained model. Hence, validation data
#'  must be provided here for \code{MOGONET} so that the function can combine
#'  them with the training data and also return the predicted labels. While
#'  for other algorithms, the validation data label prediction can be obtained
#'  via the function \code{mainpredict}, but \code{MOGONET} can't.
#'@param num_epoch_pretrain Special parameter for \code{MOGONET} and defines
#'  the epoch number for its pretraining process. Default is 500.
#'@param num_epoch Special parameter for \code{MOGONET} and defines the epoch
#'  number for its training process. Default is 2500.
#'@param adj_parameter Special parameter for \code{MOGONET} and defines the
#'  the average number of edges per node that are retained in the adjacency
#'  matrix used for graph convolutional networks (GCNs) construction. Default
#'  is 10.
#'@param dim_he_list Special parameter for \code{MOGONET} and is to define the
#'  node number of each hidden layer of the GCN networks. Need to be a vector
#'  with numbers as elements, such as \code{c(400, 400, 200)}, so that in each
#'  GCN network in \code{MOGONET}, 3 hidden layers will be set up. One is with
#'  400 nodes, while the others are with 400 and 200 nodes. The default value
#'  is \code{c(400, 400, 200)}.
#'@param lr_e_pretrain Special parameter for \code{MOGONET} and used to define
#'  the learning rate of the GCN networks for the single-omics data at their
#'  pretraining stage. Default value is 1e-3.
#'@param lr_e Special parameter for \code{MOGONET} and defines the learning
#'  rate of the GCN networks for the single-omics data at the training stage.
#'  Default value is 5e-4.
#'@param lr_c Special parameter for \code{MOGONET} and defines the learning
#'  rate of the view correlation discovery network (VCDN) to aggregate the GCN
#'  network results. Default is 1e-3.
#'@param calibrationmethod Calibration method. Can be one of "LR" (logistic
#'  regression), "FLR" (Firth's regression), and "MR" (ridge regression), or
#'  their combinations as a vector. The default is NULL. Any method provided
#'  to this parameter will be used to train a calibration model.
#'@param brglm.ctrl.max.iter A parameter special for the Firth's penalized
#'  calibration. It defines the maximum iteration number for fitting the bias
#'  reduced logistic regression model.
#'@param multiomicsnames Used for multi-omics model training. In this case, a
#'  matrix should be organized using rows as samples and columns as features,
#'  and the features should come from all the omics data want to use. Then,
#'  same as the methylation data, the function can receive the matrix via the
#'  parameter \code{betas..} or via loading the data from \code{betas.path}.
#'  Then, to demonstrate which features in the matrix are from which omics,
#'  the parameter \code{multiomicsnames} need to be used to transfer a vector
#'  to the function. The element order in the vector should be the same as the
#'  feature order in the matrix. An element is the omics name of one feature.
#'  The default value is NULL and in this case the data will be treated as
#'  single-omics data, but if an omics name indication vector is provided, the
#'  data will be treated as multi-omics data.
#'@return Will return a list containing the trained raw model and calibration
#'  models. In addition, the raw model and calibration score matrixes on the
#'  training data are also included in this list.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'RFmods <- maintrain(y.. = labels, betas.. = betas, subset.CpGs = 10000, 
#'  seed = 1234, method = 'RF', calibrationmethod = c('LR', 'FLR', 'MR'), 
#'  cores = 4)
#'  
#'summary(RFmods)
#'
#'\dontrun{
#'eSVMmods <- maintrain(y.. = labels, betas.. = betas, subset.CpGs = 10000, 
#'  seed = 1234, method = 'eSVM', calibrationmethod = c('LR', 'FLR', 'MR'), 
#'  cores = 10)
#'  
#'summary(eSVMmods)
#'}
#'@export
maintrain <- function(y.. = NULL,
                      betas.. = NULL,
                      betas.path,
                      subset.CpGs = 10000,

                      seed = 1234,
                      cores = 10,
                      topfeaturenumber = 50000,

                      savefeaturenames = FALSE,

                      method = "eSVM",

                      #SCMER & limma
                      anno = NULL,

                      #SCMER
                      lasso = 3.25e-7,
                      ridge = 0,
                      n_pcs = 100,
                      perplexity = 10,
                      savefigures = FALSE,
                      pythonpath = NULL,

                      #limma
                      confoundings = NULL,
                      padjcut = 0.05,
                      xcutoff = 0.1,
                      cutnum = 10000,

                      #RF
                      ntrees = 500,
                      p = 200,

                      #SVM, eSVM, XGBoosting, GLMNET, eNeural
                      modelcv = 5,

                      #SVM, eSVM
                      C.base = 10,
                      C.min = -3,
                      C.max = -2,

                      #eSVM, eNeural, MOGONET
                      learnernum = 10,
                      minlearnersize = 1000,
                      viewstandard = NULL,
                      platform = "450K",

                      #XGBoosting
                      max_depth = 6,
                      eta = c(0.1, 0.3),
                      gamma = c(0, 0.01),
                      colsample_bytree = c(0.01, 0.02, 0.05, 0.2),
                      subsample = 1,
                      min.chwght = 1,
                      nrounds = 100, # use default to limit computational burden
                      early_stopping_rounds = 50,

                      #GLMNET
                      alpha.min. = 0,
                      alpha.max. = 1,
                      by. = 0.1,

                      #eNeural
                      predefinedhidden = NULL,
                      maxepochs = 10,

                      activation = "Rectifier",

                      momentum_start = 0,
                      rho = 0.99,

                      gridsearch = FALSE,

                      #MOGONET
                      mogonettestbetas,

                      num_epoch_pretrain = 500,
                      num_epoch = 2500,
                      adj_parameter = 10,
                      dim_he_list = c(400, 400, 200),

                      lr_e_pretrain = 1e-3,
                      lr_e = 5e-4,
                      lr_c = 1e-3,

                      #Calibration
                      calibrationmethod = NULL,

                      brglm.ctrl.max.iter = 10000,

                      multiomicsnames = NULL,
                      weighted = FALSE){

  n.rep. <- 1

  #scmerpyfile <- system.file("python", "scmerpypackage.py", package = "methylClass")
  scmerpyfile <- '/data/liuy47/nihcodes/scmerpypackage.py'

  if(gridsearch == FALSE){
    activation <- activation[1]
    momentum_start <- momentum_start[1]
    rho <- rho[1]
  }

  #mogonetpyfile <- system.file("python", "mogonet_r.py", package = "methylClass")
  mogonetpyfile <- '/data/liuy47/nihcodes/mogonet_r.py'

  test_inverval <- 50
  featurescale <- TRUE

  scale.rowsum.to.1 <- TRUE

  if(platform == 'EPIC'){
    platform <- 850
  }else if(platform == '450K'){
    platform <- 450
  }





  if(!is.null(multiomicsnames)){
    subset.CpGs <- NULL
    topfeaturenumber <- NULL
  }


  orianno <- anno

  if(is.null(y..)){

    if(exists("y")){
      y.. <- get("y", envir = .GlobalEnv)
      message("`y` label was fetched from .GlobalEnv\n")
    }else{
      stop("Please provide `y..` labels\n")
    }
  }

  if(!is.factor(y..)){

    y.. <- as.character(y..)
    freqy <- table(y..)
    freqy <- freqy[order(-freqy)]
    y.. <- factor(y.., levels = names(freqy), ordered = TRUE)

  }


  if(is.null(betas..)){

    betas.. <- readRDS(betas.path)

  }

  if(length(y..) == ncol(betas..) & length(y..) != nrow(betas..)){
    betas.. <- t(betas..)
  }

  if(!is.null(topfeaturenumber)){

    betas.. <- topvarfeatures(betasmat = betas..,
                              topfeaturenumber = topfeaturenumber)
  }

  betas <- betas..
  rm(betas..)


  if(is.null(subset.CpGs)){

    betas <- betas[,colnames(betas)]

  }else{

    if(is.null(orianno)){

      anno <- data.frame(label = as.character(y..),
                         sentrix = row.names(betas),
                         stringsAsFactors = FALSE)
    }else if(!('label' %in% colnames(orianno))){

      anno <- data.frame(label = as.character(y..[fold$train]),
                         sentrix = row.names(betas.train),
                         stringsAsFactors = FALSE)

    }else{

      if('sentrix' %in% colnames(orianno)){

        if(sum(!(orianno$sentrix %in% row.names(betas.train))) == 0){

          sharedsamples <- intersect(orianno$sentrix, row.names(betas.train))
          anno <- subset(orianno, sentrix %in% sharedsamples)
          betas.train <- betas.train[anno$sentrix, , drop = FALSE]

        }else if(sum(!(row.names(orianno) %in% row.names(betas.train))) == 0){

          sharedsamples <- intersect(row.names(orianno), row.names(betas.train))
          anno <- orianno[sharedsamples, , drop = FALSE]
          betas.train <- betas.train[row.names(anno), , drop = FALSE]

        }else{

          anno <- data.frame(label = as.character(y..[fold$train]),
                             sentrix = row.names(betas.train),
                             stringsAsFactors = FALSE)

        }

      }else if(sum(!(row.names(orianno) %in% row.names(betas.train))) == 0){

        sharedsamples <- intersect(row.names(orianno), row.names(betas.train))
        anno <- orianno[sharedsamples, , drop = FALSE]
        betas.train <- betas.train[row.names(anno), , drop = FALSE]

      }else{

        anno <- data.frame(label = as.character(y..[fold$train]),
                           sentrix = row.names(betas.train),
                           stringsAsFactors = FALSE)

      }

    }

    features <- mainfeature(y.. = as.character(y..),
                            betas.. = betas,
                            subset.CpGs = subset.CpGs,

                            cores = cores,
                            topfeaturenumber = NULL,

                            #SCMER
                            lasso = lasso,
                            ridge = ridge,
                            n_pcs = n_pcs,
                            perplexity = perplexity,
                            savefigures = savefigures,
                            pythonpath = pythonpath,

                            #limma
                            anno = anno,
                            confoundings = confoundings,
                            padjcut = padjcut,
                            xcutoff = xcutoff,
                            cutnum = cutnum)

    features <- features$features

    if(savefeaturenames == TRUE){

      filetag <- Sys.time()
      filetag <- gsub(pattern = ':', replacement = '-', x = filetag)
      filetag <- gsub(pattern = ' ', replacement = '-', x = filetag)

      featureprefix <- 'probes'
      if(is.numeric(subset.CpGs)){
        featureprefix <- paste0('Top.', subset.CpGs, '.varprobes.')
      }else if(subset.CpGs == 'SCMER'){
        featureprefix <- 'SCMERprobes.'
      }else if(subset.CpGs == 'limma'){
        featureprefix <- 'limmaprobes.'
      }

      save(features,
           file = paste0(featureprefix, filetag, '.RData')
      )

    }

    betas <- betas[,features]

  }

  #Tune & train on training set

  if(method == 'RF'){

    # Check
    if(max(p) > ncol(betas)) {stop("< Error >: maximum value of `p` [",
                                   max(p), "] is larger than available in `betas`: [",
                                   ncol(betas), "]. Please adjust the function call\n")}

    message("Start tuning on training set using RF ... @ ", Sys.time(), "\n")

    rfcv <- trainRF(y = y..,
                    betas = betas,
                    ntrees = ntrees,
                    p = p,
                    seed = seed,
                    cores = cores)

    #Calculate Misclassification Errors (ME) on test set
    err <- sum(as.character(rfcv[[1]]$predicted) != as.character(y..))/length(y..)
    message("Misclassification error: ", err, " @ ", Sys.time(), "\n")

    trainscores <- rfcv[[1]]$votes
    res <- rfcv

  }else if(method == 'SVM'){

    message("Start tuning on training set using SVM ... @ ", Sys.time(), "\n")

    svm.linearcv <- train_SVM_e1071_LK(y = y..,
                                       betas.Train = betas,
                                       seed = seed + 1,
                                       nfolds = modelcv,
                                       mc.cores = cores,
                                       C.base = C.base,
                                       C.min = C.min,
                                       C.max = C.max,
                                       weighted = weighted)

    err.svm.e1071.probs <- sum(as.character(svm.linearcv[[1]]$fitted) != as.character(y..))/length(y..)
    message("Misclassification error: ",
            err.svm.e1071.probs, " ; @ ", Sys.time(), '\n')

    trainscores <- svm.linearcv[[3]]
    res <- svm.linearcv

  }else if(method == 'XGB'){

    y.num <- as.numeric(y..)
    y.xgb <- y.num-1
    train.mtx <- betas

    xgb.train.dgC <- tryCatch({

      as(betas, "CsparseMatrix")

    }, error = function(err){

      Matrix::Matrix(betas, sparse = TRUE)

    })

    xgb.train.dgC <- as(betas, "CsparseMatrix")  #sparse mtx (supposed to be faster)
    xgb.train.l <- list(data = xgb.train.dgC, label = y.xgb)
    dtrain <- xgboost::xgb.DMatrix(data = xgb.train.l$data,
                                   label = xgb.train.l$label) #xgboost own data structure

    watchlist <- list(train = dtrain)

    message("Start xgboost-CARET tuning on training set using XGBoosting ... @ ", Sys.time(), '\n')
    xgb.train.fit.caret <- trainXGBOOST_caret_tuner(y = y..,
                                                    train.K.k.mtx = train.mtx,
                                                    K. = as.character(Sys.Date()),
                                                    k. = gsub(pattern = ':', replacement = '-', x = format(Sys.time(), '%T')),
                                                    dtrain. = dtrain,
                                                    watchlist. = watchlist,
                                                    n.CV = modelcv,
                                                    n.rep = n.rep.,
                                                    seed. = seed,
                                                    allow.parallel = TRUE,
                                                    mc.cores = cores,
                                                    max_depth. = max_depth,
                                                    eta. = eta,
                                                    gamma. = gamma,
                                                    colsample_bytree. = colsample_bytree,
                                                    subsample. = subsample,
                                                    minchwght = min.chwght,
                                                    early_stopping_rounds. = early_stopping_rounds,
                                                    nrounds. = nrounds,
                                                    objective. = "multi:softprob",
                                                    eval_metric. = "merror",
                                                    save.xgb.model = TRUE)


    err.xgb.K.k <- sum(colnames(xgb.train.fit.caret[[2]])[apply(xgb.train.fit.caret[[2]], 1, which.max)] != as.character(y..))/length(y..)
    message("Misclassification Error = ", err.xgb.K.k, '\n')

    trainscores <- xgb.train.fit.caret[[2]]
    res <- xgb.train.fit.caret


  }else if(method == 'ENet'){

    message("Start tuning on training set using ElasticNet ... @ ", Sys.time(), '\n')

    glmnetcv.tuned <- trainGLMNET(y = y..,
                                  betas = betas,
                                  seed = seed,
                                  alpha.min = alpha.min.,
                                  alpha.max = alpha.max.,
                                  by = by.,
                                  nfolds.cvglmnet = modelcv,
                                  mc.cores = cores)


    err.probs.glmnet <- sum(colnames(glmnetcv.tuned[[2]])[apply(glmnetcv.tuned[[2]], 1, which.max)] != as.character(y..))/length(y..)

    message("Misclassification error",
            "\n of ElasticNet with alpha = ",
            glmnetcv.tuned[[3]]$opt.alpha,
            " and lambda = ",
            glmnetcv.tuned[[3]]$opt.lambda,
            " setting is: ",
            err.probs.glmnet,
            " @ ", Sys.time(), '\n')

    trainscores <- glmnetcv.tuned[[2]]
    res <- glmnetcv.tuned


  }else if(method == 'eNeural'){

    message("Start tuning on training set using eNeural ... @ ", Sys.time(), '\n')

    eNeuralcv <- eNeural(x = betas,
                         y = y..,
                         cross = modelcv,
                         seednum = seed,
                         samplenum = learnernum,
                         cores = cores,

                         predefinedhidden = predefinedhidden,
                         maxepochs = maxepochs,

                         activation = activation,

                         momentum_start = momentum_start,
                         rho = rho,

                         gridsearch = gridsearch,
                         savefile = FALSE,

                         minsize = minlearnersize,
                         viewstandard = viewstandard,
                         platform = platform,
                         multiomicsnames = multiomicsnames)

    message("Predict eNeural model ... @ ",
            Sys.time(), '\n')

    err.eNeural.probs <- sum(colnames(eNeuralcv[[2]])[apply(eNeuralcv[[2]], 1, which.max)] != as.character(y..)) / length(y..)

    message("Misclassification error: ",
            err.eNeural.probs, " ; @ ", Sys.time(), '\n')

    trainscores <- eNeuralcv[[2]]
    res <- eNeuralcv



    baselearnerpath <- './baselearners'

    if(!dir.exists(baselearnerpath)){
      dir.create(baselearnerpath, showWarnings = FALSE, recursive = TRUE)
    }

    for(i in 1:length(eNeuralcv[[1]]$baselearners)){

      baselearner <- eNeuralcv[[1]]$baselearners[[i]]

      h2o::h2o.saveModel(baselearner,
                         path = baselearnerpath,
                         force = TRUE)


    }





  }else if(method == 'MOGONET'){

    message("Start tuning on training set using MOGOnet ... ", Sys.time(), '\n')

    probesplitres <- probesplit(traindat = betas,
                                testdat = mogonettestbetas,
                                samplenum = learnernum,
                                seednum = seed,
                                minsize = minlearnersize,
                                viewstandard = viewstandard,
                                platform = platform,
                                multiomicsnames = multiomicsnames)

    scoreses <- mogonet(traingroupdatlist = probesplitres$traingroupdatlist,
                        testgroupdatlist = probesplitres$testgroupdatlist,
                        trainlabels = y..,
                        testlabels = NULL,

                        pythonpath = pythonpath,
                        mogonetpyfile = mogonetpyfile,
                        K = NULL,
                        k = NULL,

                        num_epoch_pretrain = num_epoch_pretrain,
                        num_epoch = num_epoch,
                        adj_parameter = adj_parameter,
                        dim_he_list = dim_he_list,

                        lr_e_pretrain = lr_e_pretrain,
                        lr_e = lr_e,
                        lr_c = lr_c,
                        seednum = seed,
                        test_inverval = test_inverval)




    trainscores <- scoreses$trainres

    err <- sum(colnames(trainscores)[apply(trainscores, 1, which.max)] != as.character(y..)) / length(y..)

    message("Misclassification error: ", err, " ; @ ", Sys.time(), '\n')

    res <- scoreses

  }else{

    message("Start tuning on training set using eSVM ... @ ", Sys.time(), '\n')

    svm.linearcv <- train_eSVM(y = y..,
                               betas.Train = betas,
                               seed = seed + 1,
                               mc.cores = cores,
                               nfolds = modelcv,
                               C.base = C.base,
                               C.min = C.min,
                               C.max = C.max,
                               samplenum = learnernum,
                               mod.type = "C-classification",
                               minsize = minlearnersize,
                               viewstandard = viewstandard,
                               platform = platform,
                               multiomicsnames = multiomicsnames,
                               featurescale = featurescale,
                               weighted = weighted)

    message("Predict eSVM model with tuned cost (C) parameter ... ",
            "\n Note: If the training set was scaled by eSVM (done by default),",
            " the new data is scaled accordingly using scale and center of the training data.\' ... @ ",
            Sys.time(), '\n')


    err.eSVM.probs <- sum(colnames(svm.linearcv[[2]])[apply(svm.linearcv[[2]], 1, which.max)] != as.character(y..)) / length(y..)

    message("Misclassification error: ",
            err.eSVM.probs, " ; @ ", Sys.time(), '\n')

    trainscores <- svm.linearcv[[2]]
    res <- svm.linearcv

  }

  rawres <- sub_performance_evaluator(probs.l = list(trainscores),
                                      y.. = y..,
                                      scale.rowsum.to.1 = scale.rowsum.to.1)

  message("Multiclass AUC (Hand&Till 2001) (", method, ") raw scores: ", rawres$auc.HandTill, "\n")
  message("Brier score (BS) (", method, ") raw scores: ", rawres$brier, "\n")
  message("Multiclass log loss (LL) (", method, ") raw scores: ", rawres$mlogloss, "\n")

  reslist <- list(mod = res,
                  rawscores = trainscores)

  if('LR' %in% calibrationmethod){

    message("Training calibration model using Platt scaling by LR @ ", Sys.time(), "\n")

    # Platt scaling with LR
    probs.platt.diagn.l <- list()
    platt.calfits <- list()

    c <- 1
    for(c in seq_along(colnames(trainscores))){

      diagn <- colnames(trainscores)[c]

      platt.calfit <- subfunc_Platt_train_calibration(y.i.j.innertest = y..,
                                                      scores.i.j.innertest = trainscores,
                                                      diagn.class = diagn)

      probs.platt.diagn.l[[c]] <- subfunc_Platt_fit_testset(scores.i.0.outertest = trainscores,
                                                            calib.model.Platt.diagn.i = platt.calfit,
                                                            diagn.class = diagn)

      platt.calfits[[c]] <- platt.calfit

    }

    probs.lr <- do.call(cbind, probs.platt.diagn.l)
    # Rename columns
    colnames(probs.lr) <- colnames(trainscores)

    errp <- sum(colnames(probs.lr)[apply(probs.lr, 1, which.max)] != y..)/length(y..)
    message("Misclassification error of Platt-LR-calibrated (", method, ") probabilities: ", errp, "\n")


    lrres <- sub_performance_evaluator(probs.l = list(probs.lr),
                                       y.. = y..,
                                       scale.rowsum.to.1 = scale.rowsum.to.1)

    message("Multiclass AUC (Hand&Till 2001) of Platt-LR-calibrated (", method, ") probabilities: ",
            lrres$auc.HandTill, "\n")
    message("Brier score (BS) of Platt-LR-calibrated (", method, ") probabilities: ",
            lrres$brier, "\n")
    message("Multiclass log loss (LL) of Platt-LR-calibrated (", method, ") probabilities: ",
            lrres$mlogloss, "\n")

    reslist$platt.calfits <- platt.calfits
    reslist$probs.lr <- probs.lr

  }

  if('FLR' %in% calibrationmethod){

    message("Training calibration model using Firth's penalized LR @ ", Sys.time(), "\n")

    # Platt scaling with Firth's penalized LR (FLR)
    probs.platt.firth.brglm.l <- list()
    platt.brglm.calfits <- list()

    c <- 1
    for(c in seq_along(colnames(trainscores))){

      diagn <- colnames(trainscores)[c]

      platt.brglm.calfit <- subfunc_Platt_train_calibration_Firth(y.i.j.innertest = y..,
                                                                  scores.i.j.innertest = trainscores,
                                                                  diagn.class = diagn,
                                                                  brglm.control.max.iteration = brglm.ctrl.max.iter)

      probs.platt.firth.brglm.l[[c]] <- subfunc_Platt_fit_testset_Firth(scores.i.0.outertest = trainscores,
                                                                        calib.model.Platt.Firth.diagn.i = platt.brglm.calfit,
                                                                        diagn.class = diagn)

      platt.brglm.calfits[[c]] <- platt.brglm.calfit

    }

    probs.flr <- do.call(cbind, probs.platt.firth.brglm.l)
    colnames(probs.flr) <- colnames(trainscores)

    errp <- sum(colnames(probs.flr)[apply(probs.flr, 1, which.max)] != y..)/length(y..)
    message("Misclassification error of Platt-FLR-calibrated (", method, ") probabilities : ", errp, "\n")

    flrres <- sub_performance_evaluator(probs.l = list(probs.flr),
                                        y.. = y..,
                                        scale.rowsum.to.1 = scale.rowsum.to.1)

    message("Multiclass AUC (Hand&Till 2001) of Platt-FLR-calibrated (", method, ") probabilities: ",
            flrres$auc.HandTill, "\n")
    message("Brier score (BS) of Platt-FLR-calibrated (", method, ") probabilities: ",
            flrres$brier, "\n")
    message("Multiclass log loss (LL) of Platt-FLR-calibrated (", method, ") probabilities: ",
            flrres$mlogloss, "\n")


    reslist$platt.brglm.calfits <- platt.brglm.calfits
    reslist$probs.flr <- probs.flr

  }

  if('MR' %in% calibrationmethod){

    message("Training the multinomial ridge (MR) calbriation model @ ", Sys.time(), "\n")
    # Fit multinomial ridge regression (MR) model
    set.seed(seed, kind = "default")

    if(cores > 1){

      parallel.cv.glmnet <- TRUE

      threads <- parallel::detectCores()
      cl <- parallel::makeCluster(min(cores, threads))

      doParallel::registerDoParallel(cl)


    }else{

      parallel.cv.glmnet <- FALSE

    }

    suppressWarnings(
      cv.calfit <- glmnet::cv.glmnet(y = y..,
                                     x = trainscores,
                                     family = "multinomial",
                                     type.measure = "mse",
                                     alpha = 0,
                                     nlambda = 100,
                                     lambda.min.ratio = 10^-6,
                                     parallel = parallel.cv.glmnet)
    )

    if(cores > 1){

      parallel::stopCluster(cl)

      unregister_dopar()

    }

    probs.mr <- predict(cv.calfit$glmnet.fit,
                        newx = trainscores,
                        type = "response",
                        s = cv.calfit$lambda.1se)[,,1]

    errp <- sum(colnames(probs.mr)[apply(probs.mr, 1, which.max)] != y..)/length(y..)
    message("Misclassification error of MR-calibrated (", method, ") probabilities: ", errp, "\n")


    mrres <- sub_performance_evaluator(probs.l = list(probs.mr),
                                       y.. = y..,
                                       scale.rowsum.to.1 = scale.rowsum.to.1)

    message("Multiclass AUC (Hand&Till 2001) of MR-calibrated (", method, ") probabilities: ",
            mrres$auc.HandTill, "\n")
    message("Brier score (BS) of MR-calibrated (", method, ") probabilities: ",
            mrres$brier, "\n")
    message("Multiclass log loss (LL) of MR-calibrated (", method, ") probabilities: ",
            mrres$mlogloss, "\n")


    reslist$glmnet.calfit <- cv.calfit
    reslist$probs.mr <- probs.mr

    glmnet.calfit <- cv.calfit
    rm(cv.calfit)
  }

  return(reslist)

}



#'Predict sample labels
#'
#'Predict sample labels using the trained model
#'
#'@param newdat The data with sample labels need to be predicted. Each raw is
#'  a sample and each column is a feature.
#'@param mod The raw model trained by the function \code{maintrain}. It is in
#'  the slot named "mod" of the result retured by \code{maintain}.
#'@param calibratemod The calibration model trained by \code{maintrain}, and
#'  it can be NULL if only want to predict the labels using the raw model. The
#'  default is NULL. If want to provide, it can be obtained from the result of
#'  \code{maintrain}. The slot "platt.calfits" contains the logistic model,
#'  the slot "platt.brglm.calfits" contains the Firth's model, and the slot
#'  "glmnet.calfit" contains the ridge model. Choose one of them to transfer
#'  to this parameter and the calibration model will be used to predict the
#'  sample labels.
#'@param y Not necessary but if the labels of the new samples are actually
#'  known, they can transfer to this parameter as a vector and then they will
#'  be used to compare with the predicted labels to evaluation the prediction
#'  performance.
#'@param cores The threads number need to do parallization computation and it
#'  only works for eSVM and eNeural prediction, but even no parallization is
#'  used here, the prediction speed is still not slow. Default is 1.
#'@param baselearnerpath Special for prediction using the eNeural model. This
#'  is the absolute path saving the base learner neural network model files.
#'  The function \code{maintrain} will create a folder in the working path
#'  with the base learner neural network model files saved there when eNeural
#'  model is trained and the path of this folder need to be provided to the
#'  parameter \code{baselearnerpath} here to predict the labels using eNeural.
#'@return Will return a list containing the score matrix for the new samples,
#'  and the final predicted labels in a vector. These results from the raw
#'  model will always be contained in the list, and if a calibration model is
#'  also provided to the function, the probability matrix and the labels from
#'  the calibration model will also be in the list.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'RFmods <- maintrain(y.. = labels, betas.. = betas, subset.CpGs = 10000, 
#'  seed = 1234, method = 'RF', calibrationmethod = c('LR', 'FLR', 'MR'), 
#'  cores = 4)
#'
#'RFpres <- mainpredict(newdat = betas, mod = RFmods$mod, 
#'  calibratemod = RFmods$glmnet.calfit, y = NULL, cores = 1)
#'
#'\dontrun{
#'eSVMmods <- maintrain(y.. = labels, betas.. = betas, subset.CpGs = 10000, 
#'  seed = 1234, method = 'eSVM', calibrationmethod = c('LR', 'FLR', 'MR'), 
#'  cores = 10)
#'  
#'eSVMpres <- mainpredict(newdat = betas, mod = eSVMmods$mod, 
#'  calibratemod = eSVMmods$glmnet.calfit, y = NULL, cores = 1)
#'}
#'@export
mainpredict <- function(newdat,
                        mod,
                        calibratemod = NULL,
                        y = NULL,
                        cores = 1,
                        baselearnerpath = NULL){


  if('importance' %in% names(mod[[1]])){

    newdat <- newdat[, match(rownames(mod[[1]]$importance),
                             colnames(newdat))]

    scores <- tryCatch({
      predict(mod[[1]],
              newdat,
              type = "prob")
    }, error = function(err){
      NULL
    })

    if(is.null(scores)){
      library(randomForest)
      scores <- predict(mod[[1]],
                        newdat,
                        type = "prob")

    }


  }else if('SV' %in% names(mod[[1]])){

    scores.pred.svm.e1071.obj <- e1071:::predict.svm(object = mod[[1]],
                                                     newdata = newdat[,names(mod[[1]]$x.scale[[1]]),drop = FALSE],
                                                     probability = TRUE,
                                                     decisionValues = TRUE)

    scores <- attr(scores.pred.svm.e1071.obj, "probabilities")

  }else if('handle' %in% names(mod[[1]])){

    scores.pred.xgboost.vec.test <- tryCatch({
      predict(object = mod[[1]],
              newdata = newdat,
              ntreelimit = mod[[1]]$best_iteration,
              outputmargin = FALSE)
    }, error = function(err){
      NULL
    })

    if(is.null(scores.pred.xgboost.vec.test)){
      library(xgboost)
      scores.pred.xgboost.vec.test <- predict(object = mod[[1]],
                                              newdata = newdat,
                                              ntreelimit = mod[[1]]$best_iteration,
                                              outputmargin = FALSE)

    }

    scores <- matrix(scores.pred.xgboost.vec.test,
                     nrow = nrow(newdat),
                     byrow = TRUE)

    rownames(scores) <- rownames(newdat)
    colnames(scores) <- colnames(mod[[2]])

  }else if('glmnet.fit' %in% names(mod[[1]])){

    scores <- tryCatch({
      predict(mod[[1]],
              newx = newdat,
              type="response")[,,1]
    }, error = function(err){
      NULL
    })

    if(is.null(scores)){
      library(glmnet)
      scores <- predict(mod[[1]],
                        newx = newdat,
                        type="response")[,,1]


    }



  }else if('baselearners' %in% names(mod[[1]])){

    if(grepl(pattern = 'Neural', x = names(mod[[1]]$baselearners)[1])){

      scores <- eNeuralpredict(eNeuralmod = mod[[1]],
                               x = newdat,
                               cores = cores,
                               baselearnerpath = baselearnerpath)
    }else if('trainres' %in% names(mod[[1]])){

      scores <- mod[[1]]$testres

    }else{

      scores <- eSVMpredict(eSVMmod = mod[[1]],
                            x = newdat,
                            cores = cores)

    }


  }else{

    scores <- eSVMpredict(eSVMmod = mod[[1]],
                          x = newdat,
                          cores = cores)

  }


  rawpres <- colnames(scores)[apply(scores, 1, which.max)]

  res <- list(scores = scores,
              rawpres = rawpres)


  if(!is.null(calibratemod)){


    if('glmnet.fit' %in% names(calibratemod) &
       'lambda.1se' %in% names(calibratemod)){

      library(glmnet)
      probs <- predict(calibratemod$glmnet.fit,
                       newx = scores,
                       type = "response",
                       s = calibratemod$lambda.1se)[,,1]

      if(is.vector(probs)){
        classes <- names(probs)
        probs <- matrix(probs, nrow = 1)
        row.names(probs) <- row.names(newdat)
        colnames(probs) <- classes
      }

    }else{

      probs.l <- list()

      for(c in 1:ncol(scores)){

        calmod <- calibratemod[[c]]

        diagn <- colnames(scores)[c]
        probs.l[[c]] <- subfunc_Platt_fit_testset(scores,
                                                  calmod,
                                                  diagn)

      }

      probs <- do.call(cbind, probs.l)
      colnames(probs) <- colnames(scores)
      rm(probs.l)

    }

    pres <- colnames(probs)[apply(probs, 1, which.max)]

    res$probs <- probs
    res$pres <- pres

  }


  if(!is.null(y)){

    rawerr <- sum(rawpres != y)/length(y)
    message("Misclassification raw error: ", rawerr, " @ ", Sys.time(), "\n")


    rawres <- sub_performance_evaluator(probs.l = list(scores),
                                        y.. = y,
                                        scale.rowsum.to.1 = TRUE)

    message("Multiclass AUC (Hand&Till 2001) raw scores: ", rawres$auc.HandTill, "\n")
    message("Brier score (BS) raw scores: ", rawres$brier, "\n")
    message("Multiclass log loss (LL) raw scores: ", rawres$mlogloss, "\n")

    res$rawres <- rawres


    if(!is.null(calibratemod)){

      err <- sum(pres != y)/length(y)
      message("Misclassification error: ", err, " @ ", Sys.time(), "\n")


      calres <- sub_performance_evaluator(probs.l = list(probs),
                                          y.. = y,
                                          scale.rowsum.to.1 = TRUE)

      message("Multiclass AUC (Hand&Till 2001) calibrated scores: ", calres$auc.HandTill, "\n")
      message("Brier score (BS) calibrated scores: ", calres$brier, "\n")
      message("Multiclass log loss (LL) calibrated scores: ", calres$mlogloss, "\n")

      res$calires <- calres

    }
  }

  return(res)

}






