
#other functions######


#'Perform DBSCAN on beta value matrix
#'
#'Perform density-based spatial clustering of applications with noise (DBSCAN)
#'on beta value matrix
#'
#'@param y The true labels of the samples. Can be a vector, factor, or NULL.
#'  If it is a vector or factor, each element is a label for a sample and the
#'  element order in it should be the same as the sample order in the sample
#'  data provided by the parameter \code{betas}. This is necessary only if the
#'  DBSCAN clustering is based on features selected via the \code{limma}, and
#'  in other cases, it can be set as NULL. Default is NULL.
#'@param betas The beta value matrix of the samples. Each row is one sample
#'  and each column is one feature.
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
#'@param seed The tSNE step before DBSCAN clustering involve a random process
#'  and a random seed is needed to be set here to ensure the result can be
#'  repeated. Default value is 1234.
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
#'  sample distance matrix during the \code{SCMER} selection. In addition, it
#'  also defines the number of priciple components need to do tSNE before the
#'  DBSCAN clustering. Default value is 100.
#'@param perplexity Perplexity of tSNE modeling. Default is 10.
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
#'@param eps Size (radius) of the epsilon neighborhood set for DBSCAN, default
#'  is 3.5
#'@param minPts Number of minimum points required in the eps neighborhood for
#'  core points (including the point itself) in DBSCAN. Default is 5.
#'@param legendcolumn The column number of the figure legend for the scatter
#'  plot generated. Default is 2.
#'@return Will return a beta value matrix containing the samples attributed to
#'  different DBSCAN clusters and with the ones out of the clusters removed,
#'  and also a data frame annotating the cluster IDs of these samples will be
#'  returned with the beta value matrix.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'dbscanres <- clustering(y = labels, betas = betas, subset.CpGs = 10000, 
#'  seed = 1234, cores = 4, topfeaturenumber = 50000, 
#'  eps = 3.5, minPts = 5, legendcolnum = 1)
#'@export
clustering <- function(y = NULL,
                       betas,
                       subset.CpGs = 10000,

                       seed = 1234,
                       cores = 10,
                       topfeaturenumber = 50000,

                       #SCMER
                       lasso = 3.25e-7,
                       ridge = 0,
                       n_pcs = 100,
                       perplexity = 10,
                       pythonpath = NULL,

                       #limma
                       anno = NULL,
                       confoundings = NULL,
                       padjcut = 0.05,
                       xcutoff = 0.1,
                       cutnum = 10000,

                       eps = 3.5,
                       minPts = 5,
                       legendcolnum = 2){

  scmerpyfile <- system.file("python", "scmerpypackage.py", package = "methylClass")
  #scmerpyfile <- '/data/liuy47/nihcodes/scmerpypackage.py'


  if(!is.null(y)){
    if(!is.factor(y)){
      y <- as.character(y)
      freqy <- table(y)
      freqy <- freqy[order(-freqy)]
      y <- factor(y, levels = names(freqy), ordered = TRUE)
    }
  }


  featureres <- mainfeature(y.. = y,
                            betas.. = betas,
                            subset.CpGs = subset.CpGs,

                            cores = cores,
                            topfeaturenumber = topfeaturenumber,

                            #SCMER
                            lasso = lasso,
                            ridge = ridge,
                            n_pcs = n_pcs,
                            perplexity = perplexity,
                            savefigures = FALSE,
                            pythonpath = pythonpath,

                            #limma
                            anno = anno,
                            confoundings = confoundings,
                            padjcut = padjcut,
                            xcutoff = xcutoff,
                            cutnum = cutnum
  )

  subbetas <- t(featureres$betas)

  #PCA for variable probes
  pcares <- prcomp(t(subbetas))

  limit <- min(ncol(pcares$x), n_pcs)

  plotdat <- pcares$x[,1:limit]

  #library(Rtsne)
  set.seed(seed) # Set a seed if you want reproducible results
  tsne_out <- Rtsne::Rtsne(plotdat, dims = 2,
                           perplexity = perplexity,
                           max_iter = 1000)

  if(!is.null(y)){
    tsnedat <- data.frame(tSNE1 = tsne_out$Y[,1], tSNE2 = tsne_out$Y[,2],
                          Response = y)
  }else{
    tsnedat <- data.frame(tSNE1 = tsne_out$Y[,1], tSNE2 = tsne_out$Y[,2])
  }


  res <- dbscan::dbscan(x = tsne_out$Y, eps = eps, minPts = minPts)

  plotdat <- tsnedat
  row.names(plotdat) <- colnames(subbetas)

  if(!is.null(y)){

    plotdat$Responseidx <- factor(as.numeric(as.factor(plotdat$Response)))
    plotdat$Responsepair <- paste0(plotdat$Responseidx, ':', plotdat$Response)
    plotdat$Responsepair <- factor(x = plotdat$Responsepair,
                                   level = unique(plotdat$Responsepair[order(plotdat$Responseidx)]),
                                   ordered = TRUE)

  }


  plotdat$Cluster <- res$cluster


  plotdat$Cluster <- factor(plotdat$Cluster, ordered = TRUE)

  samplenum <- nrow(plotdat)

  if(!is.null(y)){
    labelnum <- length(unique(plotdat$Responseidx))
    subtitle <- paste0('(', samplenum, ' samples with ', labelnum, ' classes)')


  }else{
    subtitle <- paste0('(', samplenum, ' samples)')
    plotdat$Responsepair <- plotdat$Responseidx <- ''

  }


  p <- ggplot2::ggplot(data = plotdat, mapping = ggplot2::aes(x = tSNE1,
                                                              y = tSNE2,
                                                              color = Responsepair,
                                                              label = Responseidx))


  if(!is.null(y)){

    print(

      p + ggplot2::geom_point(size = 3) +
        ggplot2::xlab('tSNE1') +
        ggplot2::ylab('tSNE2') +
        ggplot2::ggtitle(paste0('tSNE for top ', limit, ' PCs'),
                         subtitle = subtitle) +

        ggplot2::geom_text(hjust = 0, vjust = 0, size = 4) +

        ggplot2::scale_color_manual(values = scales::hue_pal()(labelnum)) +
        ggplot2::theme_bw() +
        ggplot2::guides(color = ggplot2::guide_legend(ncol = legendcolnum)) +
        ggplot2::theme(legend.title = ggplot2::element_blank())

    )

  }else{

    print(

      p + ggplot2::geom_point(size = 3) +
        ggplot2::xlab('tSNE1') +
        ggplot2::ylab('tSNE2') +
        ggplot2::ggtitle(paste0('tSNE for top ', limit, ' PCs'),
                         subtitle = subtitle) +

        ggplot2::geom_text(hjust = 0, vjust = 0, size = 4) +

        ggplot2::scale_color_manual(values = c('grey')) +
        ggplot2::theme_bw() +
        ggplot2::guides(color = ggplot2::guide_legend(ncol = legendcolnum)) +
        ggplot2::theme(legend.title = ggplot2::element_blank()) +
        ggplot2::theme(legend.position = "none")

    )

  }



  labelnum <- length(unique(plotdat$Cluster))
  subtitle <- paste0('(', samplenum, ' samples with ', labelnum, ' clusters)')


  if(sum(plotdat$Cluster == 0) > 0){

    labelnum <- labelnum - 1

    if(sum(plotdat$Cluster == 0) == 1){

      subtitle <- paste0('(', samplenum, ' samples with ', labelnum,
                         ' clusters and ', sum(plotdat$Cluster == 0),
                         ' noise sample)')

    }else{

      subtitle <- paste0('(', samplenum, ' samples with ', labelnum,
                         ' clusters and ', sum(plotdat$Cluster == 0),
                         ' noise samples)')

    }



  }

  p <- ggplot2::ggplot(data = plotdat, mapping = ggplot2::aes(x = tSNE1,
                                                              y = tSNE2,
                                                              color = Cluster,
                                                              label = Cluster))
  print(

    p + ggplot2::geom_point(size = 3) +
      ggplot2::xlab('tSNE1') +
      ggplot2::ylab('tSNE2') +
      ggplot2::ggtitle('DBSCAN',
                       subtitle = subtitle) +

      ggplot2::geom_text(hjust = 0, vjust = 0, size = 4) +

      ggplot2::scale_color_manual(values = c('gray', scales::hue_pal()(labelnum))) +
      ggplot2::theme_bw() +
      ggplot2::theme(legend.position = "none")

  )


  newdat <- subset(plotdat, Cluster != 0)

  if(!is.null(y)){
    newdat$Combined <- paste0(newdat$Response, '_', newdat$Cluster)
  }else{
    newdat$Combined <- as.numeric(as.character(newdat$Cluster))
  }


  newdat$Combinedidx <- factor(as.numeric(as.factor(newdat$Combined)))

  newdat$Combinedpair <- paste0(newdat$Combinedidx, ':', newdat$Combined)
  newdat$Combinedpair <- factor(x = newdat$Combinedpair,
                                level = unique(newdat$Combinedpair[order(newdat$Combinedidx)]),
                                ordered = TRUE)

  labelnum <- length(unique(newdat$Combinedidx))
  subtitle <- paste0('(', samplenum, ' samples with ', labelnum, ' new combined clusters)')

  if(!is.null(y)){

    p <- ggplot2::ggplot(data = newdat, mapping = ggplot2::aes(x = tSNE1,
                                                               y = tSNE2,
                                                               color = Combinedpair,
                                                               label = Combinedidx))
    print(

      p + ggplot2::geom_point(size = 3) +
        ggplot2::xlab('tSNE1') +
        ggplot2::ylab('tSNE2') +
        ggplot2::ggtitle('Combined',
                         subtitle = subtitle) +

        ggplot2::geom_text(hjust = 0, vjust = 0, size = 4) +

        ggplot2::scale_color_manual(values = scales::hue_pal()(labelnum)) +
        ggplot2::theme_bw() +
        ggplot2::guides(color = ggplot2::guide_legend(ncol = legendcolnum)) +
        ggplot2::theme(legend.title = ggplot2::element_blank())

    )


  }



  newbetas <- betas[row.names(newdat),] #With noise samples removed
  annodata <- newdat
  if(!is.null(y)){

    annodata <- annodata[c('tSNE1', 'tSNE2', 'Response', 'Cluster', 'Combined')]
    annodata$Response <- factor(annodata$Response)
    newy <- annodata$Response
    y <- newy

  }else{
    annodata <- annodata[c('tSNE1', 'tSNE2', 'Cluster')]
  }


  betas <- newbetas
  anno <- annodata


  clusters <- anno$Cluster
  clusters <- paste0('Cluster', clusters)
  clusters <- factor(x = clusters,
                     levels = paste0('Cluster', seq(1, length(unique(anno$Cluster)), 1)),
                     ordered = TRUE)
  if(!is.null(y)){
    anno$Response <- y
  }

  anno$Cluster <- clusters

  res <- list(betas = betas,
              anno = anno)

  return(res)

}




locienrich <- function(sub,
                       probenames,
                       dataname,
                       types){

  subsub <- subset(sub, Probe %in% probenames)
  subsub <- subsub[subsub[,ncol(subsub)] %in% names(types),]
  subsubtypes <- table(subsub[,ncol(subsub)])

  for(i in 1:length(types)){
    singletype <- names(types)[i]
    othertypes <- names(types)[-i]

    a11 <- sum(subsubtypes[singletype])
    if(is.na(a11)){
      a11 <- 0
    }
    a12 <- sum(types[singletype])

    othertypes1 <- intersect(names(subsubtypes), othertypes)
    a21 <- sum(subsubtypes[othertypes1])
    othertypes2 <- intersect(names(types), othertypes)
    a22 <- sum(types[othertypes2])

    fishermat <- matrix(c(a11, a12, a21, a22), byrow = TRUE, nrow = 2)
    fisherp <- fisher.test(fishermat)$p.val
    fisherp <- signif(fisherp, 3)

    odds <- (a11/a21)/(a12/a22)

    percent <- a11/(a11 + a21)
    if(i == 1){
      fisherps <- fisherp
      oddses <- odds
      percents <- percent
    }else{
      fisherps <- c(fisherps, fisherp)
      oddses <- c(oddses, odds)
      percents <- c(percents, percent)
    }

    names(fisherps)[length(fisherps)] <- names(percents)[length(percents)] <-
      names(oddses)[length(oddses)] <- singletype
  }

  bardata <- data.frame(regionname = names(percents), percents = percents,
                        fisherps = fisherps, oddses = oddses, stringsAsFactors = FALSE)
  bardata$dataname <- dataname

  return(bardata)
}

plotdataorganize <- function(oridat,
                             alldataname){

  oridat$dir <- 'NC'
  oridat$dir[oridat$fisherps < 0.05 & oridat$oddses > 1] <- 'UP'
  oridat$dir[oridat$fisherps < 0.05 & oridat$oddses < 1] <- 'DN'

  oridat$SS <- -log10(oridat$fisherps)
  oridat$SS[oridat$oddses < 1] <- -oridat$SS[oridat$oddses < 1]

  row.names(oridat) <- 1:nrow(oridat)

  oridat$color <- 'gray'
  oridat$color[oridat$dir == 'UP'] <- 'blue'
  oridat$color[oridat$dir == 'DN'] <- 'red'

  oridat$dir[oridat$dataname == alldataname] <- ''

  return(oridat)

}



#'Attribute methylation probes to different genomic regions
#'
#'Attribute methylation probes to different genomic regions and show their
#'distribution patterns
#'
#'@param platform The platform of the probes need to be annotated. Can be set
#'  as "450K" or "EPIC". Default is "EPIC".
#'@param allprobes The background probes to perform the Fisher's exact test to
#'  check the enrichment of probes in different genomic regions. Should be a
#'  vector with the names of the background probes as elements. Default value
#'  is NULL, and in this case, all the probes provided by another parameter
#'  \code{targetprobelist} will be used as the background probes together.
#'@param targetprobelist A list with each slot recording the probe names of
#'  a specific probe group need to check the genomic region enrichment status
#'  and the names of the slots are the group names. For example, it can be a
#'  list containing 2 slots with one named "Hypermethylated" and the other as
#'  "Hypomethylated", and the probes in each group will be mapped to different
#'  genomic regions and then for each region, its enrichment for the group
#'  probes compared with the background ones will be checked using Fisher's
#'  exact test. For each probe group, a barplot will be generated to show the
#'  probe proportion of each genomic region within that group, and whether the
#'  probes in each genomic region is significantly up or down enriched will be
#'  labeled.
#'@param plotbackground If this parameter is TRUE, a barplot showing the probe
#'  proportion of each genomic region in the background will be generated, but
#'  the enrichment significance cannot be labeled. Default is FALSE.
#'@param removedup Some methylation probes can be mapped to multiple genomic
#'  island regions or gene regions and for these special ones, whether remove
#'  them from the analysis or not. Defult is TRUE, meaning to remove them.
#'@param titlesize Font size of the plot title. Default is 20.
#'@param textsize Font size of the legend title, legend text, axis label, ect.
#'  Default is 15.
#'@param face Font face of the plot.
#'@param annotextsize Font size of the annotation text in the plot. Default is
#'  4.
#'@param face Font face of the plot.
#'@return Will return a list containing 2 slots. One indicates the statistics
#'  after attributing the probes into different island regions, and the other
#'  indicates after attributing them into different gene regions, including
#'  the percentage of the probes in each region, their Fisher's exact p-value,
#'  etc. Also, the corresponding barplots will be generated.
#'@examples
#'library(methylClass)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'top1k <- mainfeature(betas.. = betas, subset.CpGs = 1000, cores = 4, 
#'  topfeaturenumber = 50000)
#'  
#'top1klist <- list(top1k = top1k$features)
#'
#'enrichres <- multipledistribution(platform = 'EPIC', 
#'  allprobes = colnames(betas), targetprobelist = top1klist, 
#'  plotbackground = FALSE, removedup = TRUE)
#'@export
multipledistribution <- function(platform = "EPIC",
                                 allprobes = NULL,
                                 targetprobelist,
                                 plotbackground = FALSE,
                                 removedup = TRUE,
                                 textsize = 15,
                                 titlesize = 20,
                                 face = NULL,
                                 annotextsize = 4){

  if(is.null(allprobes)){
    plotbackground <- FALSE
    allprobes <- unique(as.vector(unlist(targetprobelist)))
  }


  if(platform == 'EPIC'){
    platformname <- 850
  }else if(platform == '450K'){
    platformname <- 450
  }

  allprobeanno <- probeannotation(platform = platformname, finalprobes = allprobes)

  if(removedup == TRUE){

    dups <- unique(allprobeanno$Probe[duplicated(allprobeanno$Probe)])
    allprobeanno <- subset(allprobeanno, !(Probe %in% dups))
  }

  islandspos <- allprobeanno[c('Probe', 'chr', 'pos', 'strand',
                               'Islands_Name', 'Relation_to_Island')]
  tssinfo <- allprobeanno[c('Probe', 'chr', 'pos', 'strand',
                            'UCSC_RefGene_Name', 'ENTREZID', 'UCSC_RefGene_Group')]
  tssinfo$UCSC_RefGene_Group[tssinfo$UCSC_RefGene_Group == ''] <- 'Intergenic'

  islandspostype <- table(islandspos$Relation_to_Island)
  tssinfotype <- table(tssinfo$UCSC_RefGene_Group)

  alldataname <- paste0('All ', length(allprobes), ' sites analyzed')



  islandall <- locienrich(sub = islandspos,
                          probenames = allprobes,
                          dataname = alldataname,
                          types = islandspostype)

  tssinfoall <- locienrich(sub = tssinfo,
                           probenames = allprobes,
                           dataname = alldataname,
                           types = tssinfotype)

  i <- 1
  for(i in 1:length(targetprobelist)){

    targetdataname <- names(targetprobelist)[i]
    targetprobes <- targetprobelist[[i]]

    islandtop <- locienrich(sub = islandspos,
                            probenames = targetprobes,
                            dataname = targetdataname,
                            types = islandspostype)

    if(i == 1){
      islandbar <- islandtop
    }else{
      islandbar <- rbind(islandbar, islandtop)
    }


    tssinfotop <- locienrich(sub = tssinfo,
                             probenames = targetprobes,
                             dataname = targetdataname,
                             types = tssinfotype)


    if(i == 1){
      tssinfobar <- tssinfotop
    }else{
      tssinfobar <- rbind(tssinfobar, tssinfotop)
    }

    print(i)


  }

  if(plotbackground == TRUE){

    islandbar <- rbind(islandall, islandbar)
    tssinfobar <- rbind(tssinfoall, tssinfobar)
  }


  islandbar <- plotdataorganize(oridat = islandbar,
                                alldataname = alldataname)
  tssinfobar <- plotdataorganize(oridat = tssinfobar,
                                 alldataname = alldataname)


  if(plotbackground == TRUE){
    islandbar$dataname <- factor(islandbar$dataname,
                                 levels = c(alldataname, names(targetprobelist)),
                                 ordered = TRUE)
    tssinfobar$dataname <- factor(tssinfobar$dataname,
                                  levels = c(alldataname, names(targetprobelist)),
                                  ordered = TRUE)

  }else{
    islandbar$dataname <- factor(islandbar$dataname,
                                 levels = names(targetprobelist),
                                 ordered = TRUE)
    tssinfobar$dataname <- factor(tssinfobar$dataname,
                                  levels = names(targetprobelist),
                                  ordered = TRUE)
  }

  #library(ggplot2)

  xcoord6 <- c(0.6, 0.75, 0.9, 1.1, 1.25, 1.4)
  xcoord7 <- c(0.6, 0.75, 0.9, 1, 1.1, 1.25, 1.4)
  xcoord8 <- c(0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35)

  if(length(unique(islandbar$regionname)) == 6){
    xcoord <- xcoord6
  }else if(length(unique(islandbar$regionname)) == 7){
    xcoord <- xcoord7
  }else if(length(unique(islandbar$regionname)) == 8){
    xcoord <- xcoord8
  }

  p <- ggplot2::ggplot(islandbar, ggplot2::aes(x = 1,
                                               y = percents,
                                               fill = regionname))

  print(

    p + ggplot2::geom_bar(stat = 'identity', position = 'dodge') +
      ggplot2::xlab(NULL) + ggplot2::ylab('Percentage of Sites') +
      ggplot2::ggtitle(paste0('Island Region')) +
      ggplot2::scale_fill_discrete(name = 'Island Region') +

      ggplot2::facet_wrap(ggplot2::vars(dataname)) +

      ggplot2::geom_text(data = islandbar,

                         x = rep(xcoord,
                                 length(unique(islandbar$dataname))),
                         y = max(islandbar$percents)*0.9,
                         ggplot2::aes(label = dir),

                         size = annotextsize,
                         color = islandbar$color,
                         angle = 90,
                         fontface = 'italic') +

      ggplot2::theme_bw() +

      ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                     axis.ticks.x = ggplot2::element_blank(),

                     axis.title.y = ggplot2::element_text(size = textsize, face = face),
                     legend.title = ggplot2::element_text(size = textsize, face = face),
                     legend.text = ggplot2::element_text(size = textsize, face = face),
                     plot.title = ggplot2::element_text(size = titlesize, face = face),
                     plot.subtitle = ggplot2::element_text(size = textsize, face = face),
                     strip.text.x = ggplot2::element_text(size = textsize, face = face))



  )

  if(length(unique(tssinfobar$regionname)) == 6){
    xcoord <- xcoord6
  }else if(length(unique(tssinfobar$regionname)) == 7){
    xcoord <- xcoord7
  }else if(length(unique(tssinfobar$regionname)) == 8){
    xcoord <- xcoord8
  }

  p <- ggplot2::ggplot(tssinfobar, ggplot2::aes(x = 1,
                                                y = percents,
                                                fill = regionname))


  print(

    p + ggplot2::geom_bar(stat = 'identity', position = 'dodge') +
      ggplot2::xlab(NULL) + ggplot2::ylab('Percentage of Sites') +
      ggplot2::ggtitle(paste0('Gene Region')) +
      ggplot2::scale_fill_discrete(name = 'Gene Region') +

      ggplot2::facet_wrap(ggplot2::vars(dataname)) +

      ggplot2::geom_text(data = tssinfobar,

                         x = rep(xcoord,
                                 length(unique(tssinfobar$dataname))),
                         y = max(tssinfobar$percents)*0.9,
                         ggplot2::aes(label = dir),

                         size = annotextsize,
                         color = tssinfobar$color,
                         angle = 90,
                         fontface = 'italic') +

      ggplot2::theme_bw() +

      ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                     axis.ticks.x = ggplot2::element_blank(),

                     axis.title.y = ggplot2::element_text(size = textsize, face = face),
                     legend.title = ggplot2::element_text(size = textsize, face = face),
                     legend.text = ggplot2::element_text(size = textsize, face = face),
                     plot.title = ggplot2::element_text(size = titlesize, face = face),
                     plot.subtitle = ggplot2::element_text(size = textsize, face = face),
                     strip.text.x = ggplot2::element_text(size = textsize, face = face))



  )

  res <- list(islandinfo = islandbar,
              tssinfo = tssinfobar)

  return(res)


}


JvisR <- function(datlist,

                  n_components = 2,
                  metric = 'euclidean',
                  random_state = NULL,
                  Lambda = 5,

                  perplexity = 30,

                  n_neighbors = 15,
                  min_dist = 0.1,

                  scaledat = TRUE,
                  pythonpath = NULL){



  jvispyfile <- system.file("python", "jvispackage.py", package = "methylClass")

  if(!is.null(pythonpath)){

    Sys.setenv(RETICULATE_PYTHON = pythonpath)

  }

  #reticulate::use_python(pydir)

  reticulate::py_config()

  reticulate::source_python(jvispyfile)

  if(is.null(names(datlist))){

    names(datlist) <- paste0('data', 1:length(datlist))

  }

  names(datlist)[names(datlist) == ''] <- paste0('data',
                                                 c(1:length(datlist))[names(datlist) == ''])

  if(scaledat == TRUE){

    for(i in 1:length(datlist)){
      samplenames <- row.names(datlist[[i]])
      featurenames <- colnames(datlist[[i]])
      datlist[[i]] <- apply(X = datlist[[i]], MARGIN = 2, FUN = scale)
      row.names(datlist[[i]]) <- samplenames
      colnames(datlist[[i]]) <- featurenames
    }

  }

  res <- JvisPy(datlist = datlist,
                n_components = n_components,
                metric = metric,
                random_state = random_state,
                Lambda = Lambda,

                perplexity = perplexity,

                n_neighbors = n_neighbors,
                min_dist = min_dist)

  colnames(res$JSNE) <- paste0('JSNE', seq(1, ncol(res$JSNE), 1))

  if(scaledat == TRUE){
    row.names(res$JSNE) <- samplenames
  }

  colnames(res$JUMAP) <- paste0('JUMAP', seq(1, ncol(res$JUMAP), 1))

  if(scaledat == TRUE){
    row.names(res$JUMAP) <- samplenames
  }

  return(res)

}


plotvis <- function(coordat,
                    labels = NULL,
                    legendcolnum = 1){

  plotdat <- as.data.frame(coordat, stringsAsFactors = FALSE)
  samplenum <- nrow(plotdat)

  y <- labels

  if(!is.null(y)){
    labelnum <- length(unique(y))
    subtitle <- paste0('(', samplenum, ' samples with ', labelnum, ' classes)')
    plotdat$Response <- y
    plotdat$Responseidx <- factor(as.numeric(as.factor(plotdat$Response)))
    plotdat$Responsepair <- paste0(plotdat$Responseidx, ':', plotdat$Response)
    plotdat$Responsepair <- factor(x = plotdat$Responsepair,
                                   level = unique(plotdat$Responsepair[order(plotdat$Responseidx)]),
                                   ordered = TRUE)

  }else{
    subtitle <- paste0('(', samplenum, ' samples)')
    plotdat$Response <- plotdat$Responseidx <- plotdat$Responsepair <- ''

  }

  axes <- colnames(plotdat)[c(1, 2)]
  colnames(plotdat)[c(1, 2)] <- c('Component1', 'Component2')

  p <- ggplot2::ggplot(data = plotdat, mapping = ggplot2::aes(x = Component1,
                                                              y = Component2,
                                                              color = Responsepair,
                                                              label = Responseidx))

  if(!is.null(y)){

    print(

      p + ggplot2::geom_point(size = 3) +
        ggplot2::xlab(axes[1]) +
        ggplot2::ylab(axes[2]) +
        ggplot2::ggtitle(paste0(substr(x = axes[1], start = 1,
                                       stop = nchar(axes[1]) - 1)),
                         subtitle = subtitle) +
        ggplot2::scale_color_manual(values = scales::hue_pal()(labelnum)) +
        ggplot2::geom_text(hjust = 0, vjust = 0, size = 4) +
        ggplot2::theme_bw() +

        ggplot2::guides(color = ggplot2::guide_legend(ncol = legendcolnum)) +
        ggplot2::theme(legend.title = ggplot2::element_blank())

    )

  }else{

    print(

      p + ggplot2::geom_point(size = 3) +
        ggplot2::xlab(axes[1]) +
        ggplot2::ylab(axes[2]) +
        ggplot2::ggtitle(paste0(substr(x = axes[1], start = 1,
                                       stop = nchar(axes[1]) - 1)),
                         subtitle = subtitle) +
        ggplot2::scale_color_manual(values = c('grey')) +
        ggplot2::geom_text(hjust = 0, vjust = 0, size = 4) +
        ggplot2::theme_bw() +
        ggplot2::guides(color = ggplot2::guide_legend(ncol = legendcolnum)) +
        ggplot2::theme(legend.title = ggplot2::element_blank()) +
        ggplot2::theme(legend.position = "none")

    )

  }


}


#'Perform J-SNE and J-UMAP embedding on multi-omics data
#'
#'Perform J-SNE and J-UMAP embedding on multi-omics data with scatter plot
#'generated
#'
#'@param datlist A list with each element as a single-omics data matrix, and
#'  each row of this single-omics matrix is a sample, while each column is a
#'  feature. The row names are the sample names, and the column names are the
#'  feature names. The names of the whole list elements are the omics data
#'  names.
#'@param labels The true labels of the samples. Only useful when generating
#'  the scatter plots for the J-SNE and J-UMAP results. If it is a vector
#'  containing the true labels of the samples, the sample dots on the plots
#'  can be stained with different colors representing the labels. While it can
#'  be set as NULL with no influence on the J-SNE and J-UMAP results, just all
#'  the sample dots on the final plots will be colored as gray.
#'@param n_components The number of final components after dimension reduction
#'  and its default value is 2, so that the corresponding 2-D scatter plots
#'  will also be generated. If it is not 2, J-SNE and J-UMAP results will also
#'  be returned, but no plots will be drawn.
#'@param metric The type of the sample-sample distance need to be used to do
#'  J-SNE and J-UMAP. Default is 'euclidean', but can also be 'chebyshev',
#'  'minkowski', 'correlation', 'cosine', or others.
#'@param random_state The random seed number. Default is 1234.
#'@param Lambda The value of the regularization coefficient contained in the
#'  optimization target of the J-SNE and J-UMAP. Default is 5. If some of the
#'  multi-omics data contains much noise, this value should be set as a small
#'  one such as 3, so that the noisy omics data will only account for a small
#'  weight.
#'@param perplexity The perplexity parameter for the J-SNE method. It is
#'  related to the number of nearest neighbors that is used in other manifold
#'  learning algorithms. Larger datasets usually require a larger perplexity.
#'  Consider selecting a value between 5 and 50. Default is 30.
#'@param n_neighbors The size of local neighborhood used for manifold
#'  approximation for the J-UMAP method. Larger values result in more global
#'  views of the manifold, while smaller values result in more local data
#'  being preserved. Default is 15.
#'@param min_dist The effective minimum distance between embedded points, used
#'  for the J-UMAP method. Smaller values will result in a more clustered/
#'  clumped embedding where nearby points on the manifold are drawn closer
#'  together, while larger values will result in a more even dispersal of
#'  points. Default is 0.1.
#'@param scaledat For each feature in each omics data, whether it should be
#'  centered and scaled to a mean of 0 and a standard deviation of 1. Default
#'  is TRUE.
#'@param legenedcolnum The number of columns will be accounted by the legend on
#'  the scatter plot. Default is 1.
#'@param pythonpath This function is based on \code{Python}, so the directory
#'  of \code{Python} you want to use to run it should be transferred to the
#'  function via this parameter, and several \code{Python} modules need to be
#'  installed to your \code{Python} environment, including \code{__future__},
#'  \code{numba}, \code{time}, \code{locale}, \code{warnings}, \code{sklearn},
#'  \code{joblib}, \code{pkg_resources}, \code{numpy}, \code{collections},
#'  \code{pkg_resources}, \code{scipy}, and \code{pynndescent}.
#'@return A list with the J-SNE and J-UMAP coordinates for the input samples,
#'  with corresponding scatter plots generated
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'top1k <- mainfeature(betas.. = betas, subset.CpGs = 1000, cores = 4, 
#'  topfeaturenumber = 50000)
#'
#'omicslist = list(methyl = betas[,top1k$features])
#'
#'library(reticulate)
#'
#'pypath <- py_exe()
#'
#'jvisres <- mainJvisR(datlist = omicslist, labels = labels, 
#'  random_state = 1234, pythonpath = pypath)
#'@export
mainJvisR <- function(datlist,
                      labels = NULL,

                      n_components = 2,
                      metric = "euclidean",
                      random_state = 1234,
                      Lambda = 5,

                      perplexity = 30,

                      n_neighbors = 15,
                      min_dist = 0.1,

                      scaledat = TRUE,

                      legendcolnum = 1,

                      pythonpath = NULL){


  Jvisres <- JvisR(datlist = datlist,

                   n_components = n_components,
                   metric = metric,
                   random_state = random_state,
                   Lambda = Lambda,

                   perplexity = perplexity,

                   n_neighbors = n_neighbors,
                   min_dist = min_dist,

                   scaledat = scaledat,
                   pythonpath = pythonpath)

  if(n_components == 2){

    plotvis(coordat = Jvisres$JSNE,
            labels = labels,
            legendcolnum = legendcolnum)

    plotvis(coordat = Jvisres$JUMAP,
            labels = labels,
            legendcolnum = legendcolnum)

  }

  return(Jvisres)

}





summaryfeature <- function(dat, featurecolidx){

  if(!('plyr' %in% installed.packages()[,'Package'])){
    cat('Package plyr is needed to run this function\n')
    return(NULL)
  }

  calmean <- function(block){
    gene <- unique(block$gene)
    subblock <- block[-1]
    submean <- colMeans(subblock)
    submatrix <- data.frame(submean)
    submatrix <- t(submatrix)
    row.names(submatrix) <- gene
    submatrix <- as.data.frame(submatrix)
    return(submatrix)

  }

  features <- dat[,featurecolidx]
  featurefreqs <- table(features)
  unifeatures <- names(featurefreqs[featurefreqs == 1])
  mulfeatures <- names(featurefreqs[featurefreqs > 1])

  unipart <- dat[dat[,featurecolidx] %in% unifeatures,]
  mulpart <- dat[dat[,featurecolidx] %in% mulfeatures,]

  mulpart <- plyr::ddply(.data = mulpart,
                         .variables = c(names(dat)[featurecolidx]),
                         .fun = calmean)

  row.names(mulpart) <- mulpart[,featurecolidx]
  mulpart <- mulpart[-featurecolidx]

  row.names(unipart) <- unipart[,featurecolidx]
  unipart <- unipart[-featurecolidx]

  finaldat <- rbind(unipart, mulpart)
  finaldat <- as.matrix(finaldat)

  return(finaldat)

}

#'Summarize the methylation beta values of probes to genes
#'
#'Summarize the methylation beta values of probes to genes by averaging the
#'  probes located closely to the TSS of a gene.
#'
#'@param betadat A matrix recording the beta values of methylation probes for
#'  samples. Each column represents one sample and each row represents one
#'  probe. The row names are the probe names while the column names should be
#'  sample IDs.
#'@param platform The platform of the probes. Can be set as "450K", or "EPIC".
#'@param group450k850k A vector or single string. If the data is based on 450k
#'  or EPIC platform, this parameter is needed to define which probes could be
#'  considered as related to a specific gene. Only the ones located in the
#'  gene regions included in this parameter will be considered as belong to
#'  the gene. The value of this parameter can be selected from "TSS200",
#'  "TSS1500", "1stExon", "5'UTR", '3'UTR", and "Body". The default value is
#'  the vector c("TSS200", "TSS1500", "1stExon"), which means probes within
#'  these 3 regions of a gene will be attributed to the gene and their beta
#'  values will be averaged to get the gene beta value.
#'@param includemultimatch Some probes can be attributed to more than one
#'  gene. If this parameter is TRUE, these probes will be involved into the
#'  beta value calculation for all their related genes. Otherwise, these
#'  probes will be discarded, so that the beta values of all the genes are
#'  averaged only from their uniquely related probes. Default is FALSE.
#'@return A matrix recording the summarized gene beta values for samples.
#'@examples
#'library(methylClass)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'genebetas <- togene(betadat = t(betas), platform = 'EPIC', 
#'  group450k850k = 'TSS200', includemultimatch = FALSE)
#'@export
togene <- function(betadat,
                   platform = 'EPIC',
                   group450k850k = c('TSS200', 'TSS1500', '1stExon'),
                   includemultimatch = FALSE){

  if(platform == 'EPIC'){

    platform <- 850

  }else{

    platform <- 450

  }

  range27k <- 200

  if(min(betadat) < 0 | max(betadat) > 1){

    cat('Need methylation beta value to run this function and values < 0 or > 1 cannot be used\n')
    return(NULL)

  }

  beforeprobes <- row.names(betadat)

  tssinfo <- probeannotation(platform = platform, finalprobes = beforeprobes)

  if(is.null(tssinfo)){
    return(NULL)
  }

  if(platform == 27){
    if(length(range27k) == 1){
      range27k <- c(0, range27k)
    }else{
      range27k <- c(min(range27k), max(range27k))
    }

    tssinfor <- tssinfo[!is.na(tssinfo$Distance_to_TSS),]
    tssinfor <- tssinfor[(tssinfor$Distance_to_TSS >= as.numeric(min(range27k)) &
                            tssinfor$Distance_to_TSS < as.numeric(max(range27k))),]
    tssinfor <- tssinfor[,c('Probe',
                            'Symbol', 'ENTREZID')]
  }else{
    tssinfor <- tssinfo[tssinfo$UCSC_RefGene_Group %in% group450k850k,]
    tssinfor <- tssinfor[,c('Probe',
                            'UCSC_RefGene_Name', 'ENTREZID')]
  }

  tssinfor <- unique(tssinfor)

  probes <- tssinfor$Probe

  if(includemultimatch == FALSE){

    probes <- probes[probes %in% names(table(probes)[table(probes) == 1])]

  }

  tssinfor <- subset(tssinfor, Probe %in% probes)

  tssinfor <- tssinfor[order(tssinfor[,2], tssinfor[,3]),]
  row.names(tssinfor) <- 1:nrow(tssinfor)


  tssinfor$genename <- paste0(tssinfor[,2], '::', tssinfor[,3])
  tssinfor$genename <- gsub(pattern = '^::', replacement = '',
                            x = tssinfor$genename)
  tssinfor$genename <- gsub(pattern = '::$', replacement = '',
                            x = tssinfor$genename)
  tssinfor$genename <- gsub(pattern = '^NA::', replacement = '',
                            x = tssinfor$genename)
  tssinfor$genename <- gsub(pattern = '::NA$', replacement = '',
                            x = tssinfor$genename)


  betadat <- betadat[tssinfor$Probe,]
  betadat <- as.data.frame(betadat, stringsAsFactors = FALSE)
  betadat$genename <- tssinfor$genename
  betadat <- betadat[,c(ncol(betadat), 1:(ncol(betadat) - 1))]

  betadat <- summaryfeature(dat = betadat, featurecolidx = 1)

  if(is.null(betadat)){
    return(NULL)
  }

  genenames <- row.names(betadat)
  geneorders <- order(genenames)
  betadat <- betadat[geneorders,]

  return(betadat)

}



coverredtss <- function(fragments, tssradius = NULL){

  if(!('GenomicFeatures' %in% installed.packages()[,'Package'])){
    cat('Package GenomicFeatures is needed to run this function\n')
    return(NULL)
  }

  if(!('TxDb.Hsapiens.UCSC.hg19.knownGene' %in% installed.packages()[,'Package'])){
    cat('Package TxDb.Hsapiens.UCSC.hg19.knownGene is needed to run this function\n')
    return(NULL)
  }

  if(!('IRanges' %in% installed.packages()[,'Package'])){
    cat('Package IRanges is needed to run this function\n')
    return(NULL)
  }

  if(!('GenomicRanges' %in% installed.packages()[,'Package'])){
    cat('Package GenomicRanges is needed to run this function\n')
    return(NULL)
  }

  if(!('AnnotationDbi' %in% installed.packages()[,'Package'])){
    cat('Package AnnotationDbi is needed to run this function\n')
    return(NULL)
  }

  if(!('org.Hs.eg.db' %in% installed.packages()[,'Package'])){
    cat('Package org.Hs.eg.db is needed to run this function\n')
    return(NULL)
  }

  genecoords <- suppressMessages(
    GenomicFeatures::genes(TxDb.Hsapiens.UCSC.hg19.knownGene::TxDb.Hsapiens.UCSC.hg19.knownGene))
  generanges <- genecoords@ranges

  geneids <- genecoords$gene_id
  geneseqs <- genecoords@seqnames
  genestrands <- genecoords@strand

  genetssp <- generanges@start[as.vector(genestrands == '+')]
  genetssm <- generanges@start[as.vector(genestrands == '-')] +
    generanges@width[as.vector(genestrands == '-')] - 1

  tss <- rep(0, length(genecoords))
  tss[as.vector(genestrands == '+')] <- genetssp
  tss[as.vector(genestrands == '-')] <- genetssm

  tssranges <- IRanges::IRanges(start = tss, width = 1)
  tsscoords <- GenomicRanges::GRanges(seqnames = geneseqs,
                                      ranges = tssranges,
                                      strand = genestrands,
                                      gene_id = geneids)


  fragchrs <- gsub(pattern = ':.*$', replacement = '', x = fragments)
  fragcoords <- gsub(pattern = '^chr.*:', replacement = '', x = fragments)
  fragstarts <- gsub(pattern = '-.*$', replacement = '', x = fragcoords)
  fragends <- gsub(pattern = '^.*-', replacement = '', x = fragcoords)
  fragstarts <- as.numeric(fragstarts)
  fragends <-as.numeric(fragends)

  fragranges <- IRanges::IRanges(start = fragstarts, end = fragends)
  fragranges <- GenomicRanges::GRanges(seqnames = fragchrs,
                                       ranges = fragranges, strand = '*',
                                       fragmentname = fragments)

  dis <- GenomicRanges::distanceToNearest(x = tsscoords,
                                          subject = fragranges,
                                          ignore.strand = TRUE)
  disvec <- dis@elementMetadata$distance

  rangeinfo <- fragranges[dis@to,]
  geneinfo <- genecoords[dis@from,]
  rangeinfo <- as.data.frame(rangeinfo)
  names(geneinfo) <- 1:length(geneinfo)
  geneinfo <- as.data.frame(geneinfo)
  names(rangeinfo) <- paste('range', names(rangeinfo), sep = '_')
  names(geneinfo) <- paste('gene', names(geneinfo), sep = '_')
  overlap <- cbind(rangeinfo, geneinfo)
  overlap$tssdistance <- disvec
  overlap <- unique(overlap)


  genesyms <- AnnotationDbi::select(x = org.Hs.eg.db::org.Hs.eg.db,
                                    keys = overlap$gene_gene_id,
                                    columns = 'SYMBOL',
                                    keytype = 'ENTREZID')
  overlap <- cbind(overlap, genesyms)

  generes <- data.frame(frag = overlap$range_fragmentname,
                        seqnames = overlap$gene_seqnames,
                        start = overlap$gene_start,
                        end = overlap$gene_end,
                        width = overlap$gene_width,
                        strand = overlap$gene_strand,
                        geneid = overlap$ENTREZID,
                        genename = overlap$SYMBOL,
                        TSSdistance = overlap$tssdistance,
                        stringsAsFactors = FALSE)

  if(!is.null(tssradius)){
    generes <- subset(generes, TSSdistance < tssradius)
  }

  return(generes)
}

#'Summarize the beta values of probes to DNA methylation regions (DMRs)
#'
#'Cluster the probes into DNA methylation regions (DMRs) and calculate the
#'  beta values of the DMRs via averaging the probe beta values within them.
#'
#'@param betadat A matrix recording the beta values of methylation probes for
#'  samples. Each column represents one sample and each row represents one
#'  probe. The row names are the probe names while the column names should be
#'  sample IDs.
#'@param platform The platform of the probes. Can be set as "450K" or "EPIC".
#'@param maxgap An integer indicating the cutoff of probe-probe distance when
#'  clustering the probes into DNA methylation regions (DMRs). If the distance
#'  between 2 neighbor probes is less than this cutoff, they will be clustered
#'  into the same DMR. Default is 300.
#'@param TSSradius An integer defining the TSS region that will be considered
#'  when mapping the genes to DMRs. If an DMR overlaps with a gene region from
#'  \code{TSSradius} bp upstream to \code{TSSradius} bp downstream of the TSS,
#'  this gene will be attributed to this DMR. In case when one gene region
#'  overlaps with more than one DMR, it will be attributed to the DMR with the
#'  closest distance to it, so that the genes covered by the DMRs can be
#'  annotated. Default value is 1500.
#'@return A list with 3 slots. The slot named "betadat" is a matrix recording
#'  the DMR beta values of samples. The slot named "dmrprobemapping" is a
#'  data.frame recording the probes covered by each DMR and the coordinates
#'  and other information of the DMRs and probes. The slot "dmrgenemapping" is
#'  a data.frame recording the genes whose TSS regions are covered by each
#'  DMR.
#'@examples
#'library(methylClass)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'DMRbetas <- toDMR(betadat = t(betas), platform = 'EPIC', maxgap = 300,
#'  TSSradius = 1500)
#'@export
toDMR <- function(betadat,
                  platform = 'EPIC',
                  maxgap = 300,
                  TSSradius = 1500){

  if(platform == 'EPIC'){

    platform <- 850

  }else{

    platform <- 450

  }


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

  if(!('bumphunter' %in% installed.packages()[,'Package'])){
    cat('Package bumphunter is needed to run this function\n')
    return(NULL)
  }

  if(min(betadat) < 0 | max(betadat) > 1){

    cat('Need methylation beta value to run this function and values < 0 or > 1 cannot be used\n')
    return(NULL)
  }


  if(platform == 27){
    loci <- IlluminaHumanMethylation27kanno.ilmn12.hg19::Locations
    locus <- intersect(row.names(betadat), row.names(loci))
    betadat <- betadat[locus,]
    locus <- loci[locus,]
  }else if(platform == 450){
    loci <- IlluminaHumanMethylation450kanno.ilmn12.hg19::Locations
    locus <- intersect(row.names(betadat), row.names(loci))
    betadat <- betadat[locus,]
    locus <- loci[locus,]
  }else if(platform == 850){
    loci <- IlluminaHumanMethylationEPICanno.ilm10b4.hg19::Locations
    locus <- intersect(row.names(betadat), row.names(loci))
    betadat <- betadat[locus,]
    locus <- loci[locus,]
  }

  cl <- bumphunter::clusterMaker(locus$chr, locus$pos, maxGap = maxgap)

  dmrinfor <- bumphunter::regionFinder(x = betadat[,1], chr = locus$chr,
                                       pos = locus$pos, cluster = cl,
                                       maxGap = maxgap, cutoff = 0,
                                       order = FALSE)
  dmrinfor <- dmrinfor[c('cluster', 'clusterL', 'chr', 'start', 'end')]
  names(dmrinfor)[2] <- 'probenum'
  names(dmrinfor)[1] <- 'DMR'

  betadat <- as.data.frame(betadat, stringsAsFactors = FALSE)
  betadat$dmrname <- cl
  betadat <- betadat[,c(ncol(betadat), 1:(ncol(betadat) - 1))]
  betadat$dmrname <- paste0('DMR', betadat$dmrname)

  betadat <- summaryfeature(dat = betadat, featurecolidx = 1)

  if(is.null(betadat)){

    return(NULL)
  }

  dmrnames <- row.names(betadat)
  dmridces <- gsub(pattern = 'DMR', replacement = '', x = dmrnames)
  dmridces <- as.numeric(dmridces)
  dmrorders <- match(1:length(dmrnames), dmridces)
  betadat <- betadat[dmrorders,]

  #DMR-probe annotation
  probemapping <- data.frame(Probe = row.names(locus),
                             DMR = cl, stringsAsFactors = FALSE)
  probemapping <- probemapping[order(probemapping$DMR, probemapping$Probe),]
  row.names(probemapping) <- 1:nrow(probemapping)
  probemapping <- merge(probemapping, dmrinfor, by = c('DMR'))

  probepart <- probeannotation(platform = platform,
                               finalprobes = probemapping$Probe)
  probepart <- probepart[-2]
  probemapping <- merge(probemapping, probepart, by = c('Probe'))

  probemapping <- probemapping[,c(2, 4, 5, 6, 3,
                                  1, 7, 8, 9, 10)]

  names(probemapping)[c(3, 4, 5, 7, 8)] <- c('DMR_start', 'DMR_end',
                                             'Probe_num',
                                             'Probe_pos', 'Probe_strand')
  probemapping <- probemapping[order(probemapping$DMR, probemapping$Probe_pos),]
  probemapping$DMR <- paste0('DMR', probemapping$DMR)
  probemapping <- unique(probemapping)
  row.names(probemapping) <- 1:nrow(probemapping)

  #DMR-gene annotation
  dmrs <- probemapping[,c(1, 2, 3, 4)]
  dmrs <- unique(dmrs)
  row.names(dmrs) <- 1:nrow(dmrs)

  dmrfrags <- paste0(dmrs$chr, ':', dmrs$DMR_start, '-', dmrs$DMR_end)

  genemapping <- coverredtss(fragments = dmrfrags, tssradius = TSSradius)

  if(is.null(genemapping)){
    res <- list(betadat = betadat,
                dmrprobemapping = probemapping)
    return(res)
  }

  dmrs$frag <- dmrfrags
  dmrs$fragidx <- as.numeric(row.names(dmrs))
  genemapping <- merge(dmrs, genemapping, by = c('frag'))
  genemapping <- genemapping[order(genemapping$fragidx,
                                   genemapping$start,
                                   genemapping$TSSdistance,
                                   genemapping$geneid),]
  genemapping <- unique(genemapping)
  genemapping <- genemapping[c('DMR', 'chr', 'DMR_start', 'DMR_end',
                               'geneid', 'genename',
                               'start', 'end', 'strand', 'TSSdistance')]
  colnames(genemapping)[c(7, 8, 9, 10)] <- c('Gene_start', 'Gene_end',
                                             'Gene_strand', 'DMR_TSS_distance')
  row.names(genemapping) <- 1:nrow(genemapping)

  res <- list(betadat = betadat,
              dmrprobemapping = probemapping,
              dmrgenemapping = genemapping)

  return(res)

}




plotconfusion <- function(confmat = confres$confmat,
                          title,
                          font = 12){

  library(pheatmap)

  if(sum(round(confmat) != confmat) > 0){
    format <- "%.2f"
  }else{
    format <- "%.0f"
  }

  pheatmap(confmat, cluster_rows = FALSE, cluster_cols = FALSE,
           display_numbers = TRUE, number_format = format,
           main = title,
           color = c('white',
                     colorRampPalette(rainbow(600, end = 4/6),
                                      bias = 1)(600)),
           fontsize = font)


}


dbscancomp <- function(tsnedat,
                       eps = 1.75,
                       minPts = 2,
                       classlabels,
                       title,
                       font = 12,
                       plot = TRUE){

  res <- dbscan::dbscan(x = tsnedat, eps = eps, minPts = minPts)

  dbscanlabels <- res$cluster
  dbscanlabels <- paste0('Cluster', dbscanlabels)
  dbscanlabels <- factor(x = dbscanlabels,
                         levels = paste0('Cluster', seq(min(res$cluster),
                                                        max(res$cluster),
                                                        1)),
                         ordered = TRUE)

  mat <- table(classlabels, dbscanlabels)
  mat <- unclass(mat)

  if(plot == TRUE){

    print(
      plotconfusion(confmat = mat,
                    title = title,
                    font = font)
    )

  }

  return(mat)

}


clusterscreen <- function(confmatrix = confmat, cutoff = 0.8){

  confmatrix <- confmatrix[,2:ncol(confmatrix)]
  propmat <- apply(X = confmatrix, MARGIN = 2,
                   FUN = function(x){x/sum(x)})
  maxes <- apply(X = propmat, MARGIN = 2,
                 FUN = max)
  mat <- confmatrix[,maxes > cutoff]
  propmat <- propmat[,maxes > cutoff]

  kepts <- sum(mat[propmat > cutoff])

  return(kepts)
}


#'Perform grid search to find the best DBSCAN parameters able to keep samples
#'with a matching relationship between histological labels and methylation
#'clusters
#'
#'Perform grid search across the parameters epsilon and minimum cluster points
#'  to find the best DBSCAN result to keep the most samples with a matching
#'  relationship between their histological labels and methylation clusters.
#'
#'@param tsnedat A matrix recording the sample coordinates in a  2-dimensional
#'  tSNE embedding. It should contain 2 columns, and each one corresponds the
#'  sample coordinates of one dimension. The row names are the sample names.
#'@param epses The candidate epsilon parameters for DBSCAN clustering on the
#'  tSNE coordinates. It is a numeric vector to be grid searched. Default is
#'  a vector from 0.25 to 5.00, with a stepsize of 0.25.
#'@param minPtses The candidate minimum cluster points parameters for DBSCAN
#'  clustering on the tSNE coordinates. It is a numeric vector to be searched.
#'  Default is a vector from 2 to 21, with a stepsize of 1.
#'@param cutoff A number between 0 and 1. Default value is 0.8, meaning to
#'  screen for samples with a matching relationship between their histological
#'  labels and DBSCAN clusters, if a DBSCAN cluster has > 80% of its samples
#'  with the same histological label, it will be kept, while a cluster without
#'  so many dominant samples will be dropped. For the dominant clusters, the
#'  <= 20% minor samples in them will also be dropped. Thus, the kept clusters
#'  will only contain the dominant samples.
#'@param classlabels The histological labels of the samples. Can be a vector, 
#'  factor. Each element is a label for a sample and the element order in it 
#'  should be the same as the sample order in the sample data provided by the 
#'  parameter \code{tsnedat}.
#'@return Different DBSCAN parameters will generate different clusters so that
#'  after the filtering, their kept clusters and samples will be different.
#'  This function will return the parameters with the most samples reserved.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'top1k <- mainfeature(betas.. = betas, subset.CpGs = 1000, cores = 4, 
#'  topfeaturenumber = 50000)
#'
#'omicslist = list(methyl = betas[,top1k$features])
#'
#'library(reticulate)
#'
#'pypath <- py_exe()
#'
#'jvisres <- mainJvisR(datlist = omicslist, labels = labels, 
#'  random_state = 1234, pythonpath = pypath)
#'
#'optparams <- clustergrid(tsnedat = jvisres$JSNE, epses = seq(0.25, 5, 0.25), 
#'  minPtses = seq(2, 21, 1), cutoff = 0.8, classlabels = labels)
#'@export
clustergrid <- function(tsnedat,
                        epses = seq(0.25, 5, 0.25),
                        minPtses = seq(2, 21, 1),
                        cutoff = 0.8,
                        classlabels){

  for(i in 1:length(epses)){
    eps <- epses[i]
    for(j in 1:length(minPtses)){
      minPts <- minPtses[j]

      confmat <- dbscancomp(tsnedat = tsnedat,
                            eps = eps,
                            minPts = minPts,
                            classlabels = classlabels,
                            plot = FALSE)

      if(ncol(confmat) <= 2){
        break()
      }

      keptnum <- clusterscreen(confmatrix = confmat,
                               cutoff = cutoff)

      if(i == 1 & j == 1){
        keptnums <- keptnum
        epsseries <- eps
        minPtsseries <- minPts
      }else{
        keptnums <- c(keptnums, keptnum)
        epsseries <- c(epsseries, eps)
        minPtsseries <- c(minPtsseries, minPts)
      }
    }

  }

  finalkept <- max(keptnums)
  finaleps <- min(epsseries[keptnums == finalkept])
  finalminPts <- min(minPtsseries[keptnums == finalkept])

  res <- c(finalkept, finaleps, finalminPts, cutoff)
  names(res) <- c('samplenum', 'eps', 'minPts', 'cutoff')

  return(res)

}




#'Integrate the sample labels and the DBSCAN clustering results and keep the
#'  samples with a matching relationship between them
#'
#'Integrate the sample labels and the DBSCAN clustering results and keep the
#'  samples with a matching relationship between them.
#'
#'@param tsnedat A matrix recording the sample coordinates in a  2-dimensional
#'  tSNE embedding. It should contain 2 columns, and each one corresponds the
#'  sample coordinates of one dimension. The row names are the sample names.
#'@param classlabels A vector or factor recording the sample labels, and label
#'  order should correspond to the sample order in the \code{tsnedat} matrix.
#'@param eps The epsilon parameter for DBSCAN to cluster the samples according
#'  to their tSNE coordinates.
#'@param minPtses The minimum cluster points parameter for DBSCAN to cluster
#'  the samples according to their tSNE coordinates.
#'@param cutoff A float number between 0 and 1. To screen for samples with a
#'  matching relationship between their class labels and DBSCAN clusters, if a
#'  DBSCAN cluster has samples with the same class label, and their percetage
#'  is greater than this cutoff, this cluster will be kept, while a cluster
#'  without so many dominant samples will be discarded. Then, for the dominant
#'  clusters, the minor samples in them will also be dropped. Thus, the final
#'  clusters will only contain the dominant samples.
#'@return A list with 3 slots will be returned. The slot named "samplenames"
#'  contains the reserved sample names after the filtering. Another slot named
#'  "dbscanlabels" contains the DBSCAN cluster IDs for all the samples. The
#'  slot named "classlabels" contains the class labels of all the samples.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'top1k <- mainfeature(betas.. = betas, subset.CpGs = 1000, cores = 4, 
#'  topfeaturenumber = 50000)
#'
#'omicslist = list(methyl = betas[,top1k$features])
#'
#'library(reticulate)
#'
#'pypath <- py_exe()
#'
#'jvisres <- mainJvisR(datlist = omicslist, labels = labels, 
#'  random_state = 1234, pythonpath = pypath)
#'
#'optparams <- clustergrid(tsnedat = jvisres$JSNE, epses = seq(0.25, 5, 0.25), 
#'  minPtses = seq(2, 21, 1), cutoff = 0.75, classlabels = labels)
#'  
#'clusterres <- labelclusters(tsnedat = jvisres$JSNE, classlabels = labels, 
#'  eps = optparams[['eps']], minPts = optparams[['minPts']], cutoff = 0.75)
#'@export
labelclusters <- function(tsnedat,
                          classlabels,
                          eps,
                          minPts,
                          cutoff){


  res <- dbscan::dbscan(x = tsnedat, eps = eps, minPts = minPts)

  dbscanlabels <- res$cluster
  dbscanlabels <- paste0('Cluster', dbscanlabels)
  dbscanlabels <- factor(x = dbscanlabels,
                         levels = paste0('Cluster', seq(min(res$cluster),
                                                        max(res$cluster),
                                                        1)),
                         ordered = TRUE)

  confmat <- table(classlabels, dbscanlabels)
  confmat <- unclass(confmat)
  names(dbscanlabels) <- row.names(tsnedat)
  names(classlabels) <- row.names(tsnedat)


  confmatrix <- confmat[,2:ncol(confmat)]
  propmat <- apply(X = confmatrix, MARGIN = 2,
                   FUN = function(x){x/sum(x)})
  maxes <- apply(X = propmat, MARGIN = 2,
                 FUN = max)
  mat <- confmatrix[,maxes > cutoff]
  propmat <- propmat[,maxes > cutoff]
  mat[propmat < cutoff] <- 0

  idx <- which(mat > 0, arr.ind = TRUE)
  idx <- idx[order(idx[,1], idx[,2]),]

  classnames <- row.names(mat)[idx[,1]]
  clusternames <- colnames(mat)[idx[,2]]

  for(i in 1:length(classnames)){

    classname <- classnames[i]
    clustername <- clusternames[i]

    samplenames <- names(classlabels)[classlabels %in% classname &
                                        dbscanlabels %in% clustername]

    if(i == 1){
      samplenameses <- samplenames
    }else{
      samplenameses <- c(samplenameses, samplenames)
    }

  }

  samplenames <- unique(samplenameses)
  samplenames <- names(classlabels)[names(classlabels) %in% samplenames]

  res <- list(samplenames = samplenames,
              dbscanlabels = dbscanlabels,
              classlabels = classlabels)

  return(res)

}

createnames <- function(rawnames){
  
  reversenames <- FALSE
  if(sum(grepl(pattern = '-', x = rawnames)) > 0){
    
    rawnames <- gsub(pattern = '-', replacement = '___', x = rawnames)
    reversenames <- TRUE
  }
  
  
  newnames <- make.names(names = rawnames, unique = TRUE)
  
  
  newnames[newnames %in% rawnames] <- ''
  
  
  suffix <- gsub(pattern = '^.*\\.', replacement = '', x = newnames)
  
  
  suffix <- gsub(pattern = '^X[0-9].*', replacement = '', x = suffix)
  
  
  suffix[suffix %in% rawnames] <- ''
  suffix <- paste0('.', suffix)
  suffix[suffix == '.'] <- ''
  newnames <- paste0(rawnames, suffix)
  
  if(reversenames == TRUE){
    newnames <- gsub(pattern = '___', replacement = '-', x = newnames)
  }
  
  return(newnames)
  
}

#'Perform upsampling with SMOTE
#'
#'Perform upsampling on small sample classes with SMOTE (Synthetic Minority
#'  Over-sampling Technique).
#'
#'@param dat A matrix with features as columns and samples as rows.
#'@param labels A vector or factor recording the sample class labels, and the
#'  label order should match the sample order in the \code{dat} matrix.
#'@param topfeaturenumber Before performing the upsampling, the most variable
#'  features in the whole data will be selected and others will be dropped, so
#'  that the returned data will only contain these top variable features. The
#'  default value is 50000, meaning the top 50000 variable features will be
#'  selected. If want to kept all the original features, set it as NULL.
#'@param cutoff A number to define the small sample classes. Default is 10,
#'  meaning a sample class with < 10 samples will be deemed as a small sample
#'  class and upsampling will be performed on it to synthesize some simulated
#'  samples from its original samples, so that its final sample number can be
#'  equal to 10.
#'@param downsampling For the class with a sample number > \code{cutoff}. If
#'  this parameter is set as TRUE, a dnsampling will be performed on them so
#'  that their sample number will be reduced to \code{cutoff}. Default value
#'  is FALSE.
#'@param sampleseed Random seed for the sampling processes.
#'@param k The number of the nearest neighbors of a sample to synthesize its
#'  SMOTE samples. Default is 5.
#'@param adjustbetas If the synthesized samples have beta values <= 0 or >= 1,
#'  this parameter can be set as TRUE, so that these values will be replaced
#'  by the smallest and largest values in the data and also within the range
#'  of (0, 1).
#'@return A list with 2 slots. The slot named "dat" is a matrix containing all
#'  the samples after the upsampling process. The other slot named "labels" is
#'  a vector with the corresponding class labels.
#'@examples
#'library(methylClass)
#'
#'labels <- system.file('extdata', 'testlabels.rds', package = 'methylClass')
#'labels <- readRDS(labels)
#'
#'betas <- system.file('extdata', 'testbetas.rds', package = 'methylClass')
#'betas <- readRDS(betas)
#'
#'upsampledbetas <- balancesampling(dat = betas, labels = labels, 
#'  topfeaturenumber = NULL, cutoff = 100, downsampling = FALSE, 
#'  sampleseed = 1234, k = 5, adjustbetas = TRUE)
#'@export
balancesampling <- function(dat, 
                            labels, 
                            topfeaturenumber = 50000, 
                            cutoff = 10, 
                            downsampling = FALSE, 
                            sampleseed = 1234, 
                            k = 5, 
                            adjustbetas = TRUE){
  
  labellevels <- NULL
  
  labelordered <- FALSE
  
  if(is.factor(labels)){
    labellevels <- levels(labels)
    
    labelordered <- is.ordered(labels)
  }
  
  dat <- topvarfeatures(betasmat = dat, topfeaturenumber = topfeaturenumber)
  
  classsizes <- table(as.character(labels))
  
  names(labels) <- row.names(dat)
  
  i <- 1
  for(i in 1:length(classsizes)){
    
    classname <- names(classsizes)[i]
    
    classsamples <- names(labels)[labels == classname]
    
    if(length(classsamples) > cutoff){
      
      if(downsampling == TRUE){
        
        set.seed(sampleseed + i - 1)
        subsamples <- sample(x = classsamples, size = cutoff, replace = FALSE)
        sublabels <- labels[subsamples]
        subdat <- dat[subsamples, , drop = FALSE]
        
      }else{
        
        subsamples <- classsamples
        sublabels <- labels[subsamples]
        subdat <- dat[subsamples, , drop = FALSE]
        
      }
      
    }else if(length(classsamples) < cutoff){
      
      if(length(classsamples) > 1){
        
        subdat <- dat[classsamples, , drop = FALSE]
        
        knnres <- dbscan::kNN(x = subdat, k = min(k, nrow(subdat) - 1))
        knnres <- knnres$id
        knnresidx <- seq(1, nrow(knnres)*ncol(knnres), by = 1)
        knnresidxmat <- matrix(data = knnresidx, 
                               nrow = nrow(knnres), 
                               byrow = FALSE)
        row.names(knnresidxmat) <- 1:nrow(knnresidxmat)
        colnames(knnresidxmat) <- 1:ncol(knnresidxmat)
        
        
        
        set.seed(sampleseed + i - 1)
        
        sampledidx <- sample(x = knnresidx, size = cutoff - length(classsamples), replace = TRUE)
        
        smoteidx <- do.call(rbind, 
                            lapply(X = sampledidx, 
                                   FUN = function(x){which(knnresidxmat == x, arr.ind = TRUE)}))
        row.names(smoteidx) <- 1:nrow(smoteidx)
        
        set.seed(sampleseed + i - 1)
        zetas <- runif(nrow(smoteidx))
        
        
        synthesizeddat <- lapply(X = seq(1:nrow(smoteidx)), 
                                 FUN = function(x){subdat[row.names(knnres)[smoteidx[x, 1]],] + 
                                     zetas[x]*(subdat[knnres[smoteidx[x, 1], smoteidx[x, 2]],] - 
                                                 subdat[row.names(knnres)[smoteidx[x, 1]],])})
        names(synthesizeddat) <- row.names(knnres)[smoteidx[,1]]
        
        synthesizeddat <- do.call(rbind, synthesizeddat)
        
        
        subdat <- rbind(subdat, synthesizeddat)
        
        row.names(subdat) <- createnames(rawnames = row.names(subdat))
        
        sublabels <- rep(classname, nrow(subdat))
        names(sublabels) <- row.names(subdat)
        
      }else{
        
        subdat <- dat
        
        knnres <- dbscan::kNN(x = subdat, k = min(k, nrow(subdat) - 1))
        
        mindist <- min(knnres$dist[knnres$dist > 0])
        mindistidx <- which(knnres$dist == mindist, arr.ind = TRUE)
        
        difference <- subdat[knnres$id[mindistidx[1, 1], mindistidx[1, 2]],] - 
          subdat[row.names(knnres$id)[mindistidx[1, 1]],]
        
        set.seed(sampleseed + i - 1)
        zetas <- runif(cutoff - length(classsamples))
        
        synthesizeddat <- lapply(X = 1:length(zetas), 
                                 FUN = function(x){subdat[classsamples,] + zetas[x]*difference})
        
        names(synthesizeddat) <- rep(classsamples, length(synthesizeddat))
        
        synthesizeddat <- do.call(rbind, synthesizeddat)
        
        subdat <- rbind(subdat[classsamples, , drop = FALSE], synthesizeddat)
        
        row.names(subdat) <- createnames(rawnames = row.names(subdat))
        
        sublabels <- rep(classname, nrow(subdat))
        names(sublabels) <- row.names(subdat)
        
      }
      
      
      
    }else{
      
      sublabels <- labels[classsamples]
      subdat <- dat[classsamples, , drop = FALSE]
      
    }
    
    if(i == 1){
      
      subdats <- subdat
      sublabelses <- sublabels
      
    }else{
      
      subdats <- rbind(subdats, subdat)
      sublabelses <- c(sublabelses, sublabels)
      
    }
    
  }
  
  if(adjustbetas == TRUE){
    
    minvalue <- min(subdats[subdats > 0])
    maxvalue <- max(subdats[subdats < 1])
    
    subdats[subdats <= 0] <- minvalue
    subdats[subdats >= 1] <- maxvalue
    
  }
  
  if(!is.null(labellevels)){
    
    sublabelses <- factor(sublabelses, levels = labellevels, ordered = labelordered)
    
  }
  
  res <- list(dat = subdats, 
              labels = sublabelses)
  
  return(res)
  
}



