################
## getLRscore ##
################

# The distance weights are the reciprocal function
calculate_LRTG_score_V2 <- function(exprMat, distMat, annoMat, group = NULL,
                                 LRpairs, TGs, Receiver, Sender = NULL, 
                                 far.ct = 0.75, close.ct = 0.25, 
                                 downsample = FALSE)
{
  
  receBars = annoMat %>% dplyr::filter(Cluster == Receiver) %>% 
    dplyr::select(Barcode) %>% unlist() %>% as.character()
  if(is.character(Sender)){
    sendBars = annoMat %>% dplyr::filter(Cluster == Sender) %>% 
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }else{
    sendBars = annoMat %>% dplyr::filter(Cluster != Receiver) %>% 
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }
  
  Receptors = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,2]})
  Ligands = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,1]})
  
  # get exprMat of Ligand
  LigMats = lapply(TGs, function(tg){
    # print(tg)
    ligands = Ligands[[tg]]
    if(length(ligands)==1){
      lig_count = exprMat[ligands, sendBars]
      lig_count = matrix(lig_count,nrow = 1)
    }else{
      lig_count = exprMat[ligands, sendBars] %>% as.matrix()
    }
    rownames(lig_count) = LRpairs[[tg]]
    colnames(lig_count) = sendBars
    lig_count
  })
  names(LigMats) = TGs
  
  # get exprMat of Receptor
  RecMats = lapply(TGs, function(tg){
    receptors = Receptors[[tg]]
    if(length(receptors)==1){
      rec_count = exprMat[receptors, receBars]
      rec_count = matrix(rec_count,nrow = 1)
    }else{
      rec_count = exprMat[receptors, receBars] %>% as.matrix()
    }
    rownames(rec_count) = LRpairs[[tg]]
    colnames(rec_count) = receBars
    rec_count
  })
  names(RecMats) = TGs
  
  distMat = distMat[sendBars, receBars]
  
  if(!is.null(group)){
    cpMat <- get_cell_pairs(group, distMat, far.ct, close.ct)
  }else{
    cpMat <- NULL
  }
  distMat = 1/distMat
  
  t1 <- Sys.time(); message(paste0('Start at: ',as.character(t1)))
  LRs_score = lapply(TGs, function(tg){
    
    # print(tg)
    LigMat = LigMats[[tg]]
    RecMat = RecMats[[tg]]
    lr = LRpairs[[tg]] 
    
    if(is.null(cpMat)){
      
      LR_score = RecMat*(LigMat%*%distMat)
      LR_score = t(LR_score)
      colnames(LR_score) = lr
      rownames(LR_score) = receBars
      
    }else{
      
      LR_score = lapply(unique(cpMat$Receiver), function(j){
        # j = unique(cpMat$Receiver)[1]
        is <- cpMat$Sender[cpMat$Receiver == j] %>% unique()
        if(length(is)==1){
          RecMat[,j]*(LigMat[,is]*distMat[is,j])
        }else{
          RecMat[,j]*(LigMat[,is]%*%distMat[is,j])
        }
      }) %>% do.call('cbind',.) %>% t()
      colnames(LR_score) = lr
      rownames(LR_score) = unique(cpMat$Receiver)
      
    }
    LR_score
    
  })
  names(LRs_score) = TGs
  t2 <- Sys.time(); message(paste0('End at: ',as.character(t2)))
  t2-t1
  
  if(is.null(cpMat)){
    
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, receBars] })
    
  }else{
    
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, unique(cpMat$Receiver)] })
    
  }
  names(TGs_expr) = TGs
  
  # downsample
  if(length(receBars)>500){
    if(isTRUE(downsample)){
      set.seed(2021)
      
      if(is.null(cpMat)){
        keep_cell = sample(receBars, size = 500, replace = F)
      }else{
        keep_cell = sample(unique(cpMat$Receiver), size = 500, replace = F)
      }
      
      LRs_score = lapply(LRs_score, function(LR_score){ LR_score = LR_score[keep_cell,] })
      TGs_expr = lapply(TGs_expr, function(TG_count){ TG_count = TG_count[keep_cell] })
    }
  }
  
  LRTG_score = list(LRs_score = LRs_score, TGs_expr = TGs_expr)
  
  return(LRTG_score)
  
}

# The distance weights are the exponential function
calculate_LRTG_score_V3 <- function(exprMat, distMat, annoMat, group = NULL,
                                    LRpairs, TGs, Receiver, Sender = NULL, 
                                    far.ct = 0.75, close.ct = 0.25, 
                                    downsample = FALSE)
{
  
  # get Sender and Receiver
  receBars = annoMat %>% dplyr::filter(Cluster == Receiver) %>% 
    dplyr::select(Barcode) %>% unlist() %>% as.character()
  if(is.character(Sender)){
    sendBars = annoMat %>% dplyr::filter(Cluster == Sender) %>% 
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }else{
    sendBars = annoMat %>% dplyr::filter(Cluster != Receiver) %>% 
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }
  
  # get Ligand and Receptor
  Receptors = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,2]})
  Ligands = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,1]})
  
  # get exprMat of Ligand
  LigMats = lapply(TGs, function(tg){
    # print(tg)
    ligands = Ligands[[tg]]
    if(length(ligands)==1){
      lig_count = exprMat[ligands, sendBars]
      lig_count = matrix(lig_count,nrow = 1)
    }else{
      lig_count = exprMat[ligands, sendBars] %>% as.matrix()
    }
    rownames(lig_count) = LRpairs[[tg]]
    colnames(lig_count) = sendBars
    lig_count
  })
  names(LigMats) = TGs
  
  # get exprMat of Receptor
  RecMats = lapply(TGs, function(tg){
    receptors = Receptors[[tg]]
    if(length(receptors)==1){
      rec_count = exprMat[receptors, receBars]
      rec_count = matrix(rec_count,nrow = 1)
    }else{
      rec_count = exprMat[receptors, receBars] %>% as.matrix()
    }
    rownames(rec_count) = LRpairs[[tg]]
    colnames(rec_count) = receBars
    rec_count
  })
  names(RecMats) = TGs
  
  # get and transform distMat
  l.scale = 100
  distMat = distMat[sendBars, receBars]
  
  # get cell pair
  if(!is.null(group)){
    cpMat <- get_cell_pairs(group, distMat, far.ct, close.ct)
  }else{
    cpMat <- NULL
  }
  distMat = exp(-distMat^2/(2*l.scale^2))
  
  # get LRscore
  t1 <- Sys.time(); message(paste0('Start at: ',as.character(t1)))
  LRs_score = lapply(TGs, function(tg){
    # print(tg)
    LigMat = LigMats[[tg]]
    RecMat = RecMats[[tg]]
    lr = LRpairs[[tg]] 
    
    if(is.null(cpMat)){
      
      LR_score = RecMat*(LigMat%*%distMat)
      LR_score = t(LR_score)
      colnames(LR_score) = lr
      rownames(LR_score) = receBars
      
    }else{
      
      LR_score = lapply(unique(cpMat$Receiver), function(j){

        is <- cpMat$Sender[cpMat$Receiver == j] %>% unique()
        RecMat[,j]*(LigMat[,is]%*%distMat[is,j])
      }) %>% do.call('rbind',.) 
      colnames(LR_score) = lr
      rownames(LR_score) = unique(cpMat$Receiver)
      
    }
    LR_score
    
  })
  names(LRs_score) = TGs
  t2 <- Sys.time(); message(paste0('End at: ',as.character(t2)))
  t2-t1 
  
  # get expression of each TG
  if(is.null(cpMat)){
    
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, receBars] })
    
  }else{
    
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, unique(cpMat$Receiver)] })
    
  }
  names(TGs_expr) = TGs
  
  # downsample
  if(length(receBars)>500){
    if(isTRUE(downsample)){
      set.seed(2021)
      
      if(is.null(cpMat)){
        keep_cell = sample(receBars, size = 500, replace = F)
      }else{
        keep_cell = sample(unique(cpMat$Receiver), size = 500, replace = F)
      }
      
      LRs_score = lapply(LRs_score, function(LR_score){ LR_score = LR_score[keep_cell,] })
      TGs_expr = lapply(TGs_expr, function(TG_count){ TG_count = TG_count[keep_cell] })
    }
  }
  
  LRTG_score = list(LRs_score = LRs_score, TGs_expr = TGs_expr)
  
  return(LRTG_score)
  
}

# The distance weights are the constant function
calculate_LRTG_score_V4 <- function(exprMat, distMat, annoMat, group = NULL,
                                    LRpairs, TGs, Receiver, Sender = NULL, 
                                    far.ct = 0.75, close.ct = 0.25, 
                                    downsample = FALSE)
{
  
  # get Sender and Receiver
  receBars = annoMat %>% dplyr::filter(Cluster == Receiver) %>% 
    dplyr::select(Barcode) %>% unlist() %>% as.character()
  if(is.character(Sender)){
    sendBars = annoMat %>% dplyr::filter(Cluster == Sender) %>% 
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }else{
    sendBars = annoMat %>% dplyr::filter(Cluster != Receiver) %>% 
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }
  
  # get Ligand and Receptor
  Receptors = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,2]})
  Ligands = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,1]})
  
  # get exprMat of Ligand
  LigMats = lapply(TGs, function(tg){
    # print(tg)
    ligands = Ligands[[tg]]
    if(length(ligands)==1){
      lig_count = exprMat[ligands, sendBars]
      lig_count = matrix(lig_count,nrow = 1)
    }else{
      lig_count = exprMat[ligands, sendBars] %>% as.matrix()
    }
    rownames(lig_count) = LRpairs[[tg]]
    colnames(lig_count) = sendBars
    lig_count
  })
  names(LigMats) = TGs
  
  # get exprMat of Receptor
  RecMats = lapply(TGs, function(tg){
    receptors = Receptors[[tg]]
    if(length(receptors)==1){
      rec_count = exprMat[receptors, receBars]
      rec_count = matrix(rec_count,nrow = 1)
    }else{
      rec_count = exprMat[receptors, receBars] %>% as.matrix()
    }
    rownames(rec_count) = LRpairs[[tg]]
    colnames(rec_count) = receBars
    rec_count
  })
  names(RecMats) = TGs
  
  # get and transform distMat
  distMat = distMat[sendBars, receBars]
  
  # get cell pair
  if(!is.null(group)){
    cpMat <- get_cell_pairs(group, distMat, far.ct, close.ct)
  }else{
    cpMat <- NULL
  }
  distMat[sendBars, receBars] = 1
  
  # get LRscore
  t1 <- Sys.time(); message(paste0('Start at: ',as.character(t1)))
  LRs_score = lapply(TGs, function(tg){
    print(tg)
    LigMat = LigMats[[tg]]
    RecMat = RecMats[[tg]]
    lr = LRpairs[[tg]] 
    
    if(is.null(cpMat)){
      
      LR_score = RecMat*(LigMat%*%distMat)/length(sendBars)
      LR_score = t(LR_score)
      colnames(LR_score) = lr
      rownames(LR_score) = receBars
      
    }else{
      
      LR_score = lapply(unique(cpMat$Receiver), function(j){

        is <- cpMat$Sender[cpMat$Receiver == j] %>% unique()
        RecMat[,j]*(LigMat[,is]%*%distMat[is,])/length(is)
      }) %>% do.call('rbind',.) 
      colnames(LR_score) = lr
      rownames(LR_score) = unique(cpMat$Receiver)
      
    }
    LR_score
    
  })
  names(LRs_score) = TGs
  t2 <- Sys.time(); message(paste0('End at: ',as.character(t2)))
  t2-t1 
  
  # get expression of each TG
  if(is.null(cpMat)){
    
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, receBars] })
    
  }else{
    
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, unique(cpMat$Receiver)] })
    
  }
  names(TGs_expr) = TGs
  
  # downsample
  if(length(receBars)>500){
    if(isTRUE(downsample)){
      set.seed(2021)
      
      if(is.null(cpMat)){
        keep_cell = sample(receBars, size = 500, replace = F)
      }else{
        keep_cell = sample(unique(cpMat$Receiver), size = 500, replace = F)
      }
      
      LRs_score = lapply(LRs_score, function(LR_score){ LR_score = LR_score[keep_cell,] })
      TGs_expr = lapply(TGs_expr, function(TG_count){ TG_count = TG_count[keep_cell] })
    }
  }
  
  LRTG_score = list(LRs_score = LRs_score, TGs_expr = TGs_expr)
  
  return(LRTG_score)
  
}

# The distance weights follow exp(-0.5*d)/d
calculate_LRTG_score_V5 <- function(exprMat, distMat, annoMat, group = NULL,
                                    LRpairs, TGs, Receiver, Sender = NULL, 
                                    far.ct = 0.75, close.ct = 0.25, 
                                    downsample = FALSE)
{
  receBars = annoMat %>% dplyr::filter(Cluster == Receiver) %>%
    dplyr::select(Barcode) %>% unlist() %>% as.character()
  if(is.character(Sender)){
    sendBars = annoMat %>% dplyr::filter(Cluster == Sender) %>%
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }else{
    sendBars = annoMat %>% dplyr::filter(Cluster != Receiver) %>%
      dplyr::select(Barcode) %>% unlist() %>% as.character()
  }

  Receptors = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,2]})
  Ligands = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,1]})

  LigMats = lapply(TGs, function(tg){
    ligands = Ligands[[tg]]
    if(length(ligands)==1){
      lig_count = exprMat[ligands, sendBars]
      lig_count = matrix(lig_count,nrow = 1)
    }else{
      lig_count = exprMat[ligands, sendBars] %>% as.matrix()
    }
    rownames(lig_count) = LRpairs[[tg]]
    colnames(lig_count) = sendBars
    lig_count
  })
  names(LigMats) = TGs

  RecMats = lapply(TGs, function(tg){
    receptors = Receptors[[tg]]
    if(length(receptors)==1){
      rec_count = exprMat[receptors, receBars]
      rec_count = matrix(rec_count,nrow = 1)
    }else{
      rec_count = exprMat[receptors, receBars] %>% as.matrix()
    }
    rownames(rec_count) = LRpairs[[tg]]
    colnames(rec_count) = receBars
    rec_count
  })
  names(RecMats) = TGs

  distMat = distMat[sendBars, receBars]
  zero_mask <- distMat == 0
  if(any(zero_mask)){
    distMat[zero_mask] <- 1e-6
  }

  if(!is.null(group)){
    cpMat <- get_cell_pairs(group, distMat, far.ct, close.ct)
  }else{
    cpMat <- NULL
  }
  distMat = exp(-0.5 * distMat) / distMat
  distMat[!is.finite(distMat)] <- 0

  t1 <- Sys.time(); message(paste0('Start at: ',as.character(t1)))
  LRs_score = lapply(TGs, function(tg){
    LigMat = LigMats[[tg]]
    RecMat = RecMats[[tg]]
    lr = LRpairs[[tg]]

    if(is.null(cpMat)){

      LR_score = RecMat*(LigMat%*%distMat)
      LR_score = t(LR_score)
      colnames(LR_score) = lr
      rownames(LR_score) = receBars

    }else{

      LR_score = lapply(unique(cpMat$Receiver), function(j){
        is <- cpMat$Sender[cpMat$Receiver == j] %>% unique()
        if(length(is)==1){
          RecMat[,j]*(LigMat[,is]*distMat[is,j])
        }else{
          RecMat[,j]*(LigMat[,is]%*%distMat[is,j])
        }
      }) %>% do.call('cbind',.) %>% t()
      colnames(LR_score) = lr
      rownames(LR_score) = unique(cpMat$Receiver)

    }
    LR_score

  })
  names(LRs_score) = TGs
  t2 <- Sys.time(); message(paste0('End at: ',as.character(t2)))

  if(is.null(cpMat)){
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, receBars] })
  }else{
    TGs_expr = lapply(TGs, function(tg){ exprMat[tg, unique(cpMat$Receiver)] })
  }
  names(TGs_expr) = TGs

  if(length(receBars)>500){
    if(isTRUE(downsample)){
      set.seed(2021)

      if(is.null(cpMat)){
        keep_cell = sample(receBars, size = 500, replace = F)
      }else{
        keep_cell = sample(unique(cpMat$Receiver), size = 500, replace = F)
      }

      LRs_score = lapply(LRs_score, function(LR_score){ LR_score = LR_score[keep_cell,] })
      TGs_expr = lapply(TGs_expr, function(TG_count){ TG_count = TG_count[keep_cell] })
    }
  }

  LRTG_score = list(LRs_score = LRs_score, TGs_expr = TGs_expr)

  return(LRTG_score)

}

get_cell_pairs <- function(group=NULL, distMat, far.ct = 0.75, close.ct = 0.25)
{
  
  distMat_long <- reshape2::melt(distMat)
  colnames(distMat_long) <- c('Sender','Receiver','Distance')
  distMat_long$Sender <- as.character(distMat_long$Sender)
  distMat_long$Receiver <- as.character(distMat_long$Receiver)
  
  if(is.null(group)){
    respon_cellpair <- distMat_long[,1:2]
    group = 'all'
  }
  if(group == 'close') 
    respon_cellpair <- distMat_long[distMat_long$Distance <= quantile(distMat_long$Distance,close.ct),1:2]
  if(group == 'far') 
    respon_cellpair <- distMat_long[distMat_long$Distance >= quantile(distMat_long$Distance,far.ct),1:2]
  
  return(respon_cellpair)
}

####################
## train RF model ##
####################

# mian
get_pim_auto = function(trainx, trainy, ncores = 1, auto_para = TRUE, 
                         n.trees = 500, node.feature = 'sqrt', n.trys = 10, 
                         tree.method = 'variance', node.size = 5,
                         nPrem = 10, verbose = TRUE)
{
  
  # data
  LRpairs = colnames(trainx)
  data = as.data.frame(cbind(trainx, trainy))
  colnames(data) = c(paste0('LR',seq(ncol(trainx))), "Target")
  
  # get Lig, Rec
  LRTab = data.frame(LRpair = LRpairs)
  LRTab$Ligand = strsplit(LRpairs,"_") %>% do.call('rbind',.) %>% .[,1]
  LRTab$Receptor = strsplit(LRpairs,"_") %>% do.call('rbind',.) %>% .[,2]
  
  cat("\nRemove zero Target")
  zeroTarget = which(trainy==0)
  if(length(zeroTarget)>1) data = data[-zeroTarget,]
  
  cat("\nRemove zero LR")
  zeroLR = which(colSums(data[,-ncol(data)]==0)==nrow(data))
  zeroLR = c(zeroLR,which(is.na(colSums(data[,-ncol(data)]==0))))
  if(length(zeroLR)>1){
    data = data[,-zeroLR]
    LRTab = LRTab[-zeroLR,]
  }
  
  # normalize
  cat("\nNormalize data")
  data = scale(data)
  data = as.data.frame(data)
  
  # get optimal parameters of RF model
  cat("\nParameter Tuning")
  if(verbose){
    
    t1 <- Sys.time()
    cat(paste0("\nStart at: ",as.character(t1)))
    
  }
  if(auto_para == TRUE){
    
    fitControl <- trainControl(
      method = "cv",
      number = 5,
      search = 'random',
      allowParallel = TRUE)
    
    set.seed(2021)
    rfFit <- caret::train(Target ~ ., data = data, 
                          method = 'ranger', 
                          trControl = fitControl,
                          tuneLength = 20)
    
    # optimal parameter
    sel_mtry = as.numeric(rfFit$bestTune$mtry)
    sel_splitrule = as.character(rfFit$bestTune$splitrule)
    sel_min.node.size = as.numeric(rfFit$bestTune$min.node.size)
    sel_num.trees = n.trees
    
  }else{
    
    sel_mtry = n.trys
    sel_splitrule = tree.method
    sel_min.node.size = node.size
    sel_num.trees = n.trees
    rfFit <- list()
    
  }
  parameters = list(sel_mtry, sel_splitrule, sel_min.node.size, sel_num.trees)
  names(parameters) = c('mtry','splitrule','min.node.size','n.trees')
  cat(paste0("\nThe final parameters used for the model: mtry = ",sel_mtry,
             ', splitrule = ',sel_splitrule, ', min.node.size = ',sel_min.node.size, 
             ' ,num.trees = ',sel_num.trees))
  if(verbose){
    
    t2 <- Sys.time()
    message(paste0("\nEnd at: ",as.character(t2)))
    message(paste0('\nAbout ',signif(t2-t1,digits = 4),' ',units(t2-t1)))
    
  } 
  
  # train RF model
  cat("\nTrain final model")
  finalFit <- ranger(formula = Target ~ ., data = data, 
                     num.trees = sel_num.trees, seed = 2021, 
                     splitrule = sel_splitrule, mtry = sel_mtry, 
                     min.node.size = sel_min.node.size,
                     importance = 'permutation', keep.inbag = TRUE, 
                     oob.error = TRUE, num.threads = ncores)
  
  # get permutation importance based on LRpair
  cat("\nobtain variable importance")
  df_IM = finalFit$variable.importance
  df_IM = data.frame(IM = df_IM)
  df_IM$LRpair = LRTab$LRpair
  
  # get permutation importance based on Lig/Rec
  cat("\nobtain permutation importance")
  if(verbose){
    
    t1 <- Sys.time()
    message(paste0("\nStart at: ",as.character(t1)))
    
  }
  df_pIM <- lapply(seq(nPrem), function(j){
    
    forest = finalFit
    data_X=data[,-ncol(data)]
    data_y=data$Target
    rf_oob_pim(forest, data_X, data_y, LRTab)
    
  })
  df_pIM <- Reduce("+", df_pIM)/nPrem
  if(verbose){
    
    t2 <- Sys.time()
    message(paste0("\nEnd at: ",as.character(t2)))
    message(paste0('\nAbout ',signif(t2-t1,digits = 4),' ',units(t2-t1)))
    
  }
  
  result <- list(model = finalFit,
                 df_IM = df_IM,
                 df_pIM = df_pIM,
                 rfFit = rfFit)
  
  return(result)
}

# get pIM from shuffle and OOB data
rf_oob_pim = function(forest, data_X, data_y, LRTab)
{
  
  # get pred value based on OOB data
  oob_pred = rf_oob_pred(forest, data_X)
  oob_pred = as.data.frame(oob_pred)
  
  # MSE before shuffle (500 x 1)
  oob_mse_bef = lapply(seq(ncol(oob_pred)), function(k){
    
    oob_pred_k = oob_pred[,k]
    mean((oob_pred_k-data_y)^2,na.rm = T)
    
  })
  oob_mse_bef = unlist(oob_mse_bef)
  
  # get Lig/Rec
  Vars = unique(c(LRTab$Ligand,LRTab$Receptor))
  
  # MSE after shuffle (500 x len(Vars))
  oob_mse_aft = lapply(seq(length(Vars)), function(i){
    
    # shuffle one Lig/Rec
    newX = shuffle_LigRec(data_X,i,LRTab)
    
    # pred value based on OOB data after shuffle one Lig/Rec
    oob_pred_i = rf_oob_pred(forest, newX)
    oob_pred_i = as.data.frame(oob_pred_i)
    
    # MSE after shuffle one Lig/Rec
    oob_mse_aft_i = lapply(seq(ncol(oob_pred_i)), function(k){
      
      oob_pred_i_k = oob_pred_i[,k]
      mean((oob_pred_i_k-data_y)^2,na.rm = T)
      
    })
    oob_mse_aft_i = unlist(oob_mse_aft_i)
    
  })
  oob_mse_aft = do.call("cbind",oob_mse_aft)
  
  # mean MSE of each Lig/Rec
  df_pIM = apply(oob_mse_aft, 2, function(oob_mse_aft_i){ mean(oob_mse_aft_i-oob_mse_bef) })
  df_pIM = data.frame(pIM = df_pIM)
  rownames(df_pIM) = paste0('shuffle_',Vars)
  
  return(df_pIM)
  
}

# OOB data
# get pred value of OOB data from each tree(n.trees = 500 => 500 trees)
# for each tree, trainx was split into OOB data (used for evaluation) and train data (used for training RF)
rf_oob_pred <- function(forest, data_X) 
{
  
  preds <- predict(forest, data_X, predict.all=TRUE)
  oob <- forest$inbag.counts
  oob <- do.call("cbind",oob)
  oob <- oob==0
  oob[which(!oob)] = NA
  preds.oob = oob*preds$predictions 
  return (preds.oob)
  
}

# shuffle
shuffle_LigRec <- function(data_X,i,LRTab)
{
  
  Vars = unique(c(LRTab$Ligand,LRTab$Receptor))
  var <- Vars[i]
  var_ids <- union(which(LRTab$Ligand %in% var),which(LRTab$Receptor %in% var))
  for(id in var_ids) {
    data_X[,id] <- sample(x = unlist(data_X[,id]), size = nrow(data_X), replace = FALSE)
  }
  return(data_X)
  
}

###################
## calculate_auc ##
###################

## get different metrices and ROC/PRC preformance object
get_evaluate_metrics <- function(pred,label)
{
  
  find_optimal_cutoff <- function(TPR, FPR, threshold){
    
    y = TPR - FPR
    Youden_index = which.max(y)
    # optimal_threshold = threshold[Youden_index]
    return(Youden_index)
    
  }
  
  if(length(which(label==TRUE))!=0 & length(which(label==FALSE))!=0){
    
    pred <- prediction(pred, label)
    
    perf_ROC <- performance(pred, measure = "tpr", x.measure = "fpr")
    ind_ROC <- find_optimal_cutoff(perf_ROC@y.values[[1]],perf_ROC@x.values[[1]],perf_ROC@alpha.values[[1]])
    cutoff_ROC <- perf_ROC@alpha.values[[1]][ind_ROC]
    
    perf_PRC <- performance(pred, measure = "prec", x.measure = "rec")
    ind_PRC <- find_optimal_cutoff(perf_PRC@y.values[[1]],perf_PRC@x.values[[1]],perf_PRC@alpha.values[[1]])
    cutoff_PRC <- perf_PRC@alpha.values[[1]][ind_PRC]
    
    ACC <- performance(pred, measure = "acc")@y.values[[1]] %>% .[ind_ROC] %>% signif(.,4)
    ERR <- performance(pred, measure = "err")@y.values[[1]] %>% .[ind_ROC]  %>% signif(.,4)
    PPV <- performance(pred, measure = "ppv")@y.values[[1]] %>% .[ind_ROC]  %>% signif(.,4)
    MCC <- performance(pred, measure = "mat")@y.values[[1]] %>% .[ind_ROC]  %>% signif(.,4)
    
    AUC <- performance(pred, measure = "auc")@y.values[[1]] %>% signif(.,4)
    AUCPR <- performance(pred, measure = "aucpr")@y.values[[1]] %>% signif(.,4)
    
    res = list(perf_ROC = perf_ROC,
               perf_PRC = perf_PRC,
               perf_metrics = c(ROC_AUC=AUC,PRC_AUC=AUCPR,
                                ACC=ACC,ERR=ERR,PPV=PPV,MCC=MCC))
    
  }else{
    
    res <- list(perf_ROC = NA, perf_PRC = NA, 
                perf_metrics = rep(0,6))
    names(res$perf_metrics) <- c('ROC_AUC','PRC_AUC','ACC','ERR','PPV','MCC')
    
  }
  
  
  return(res)
  
}

## get points from ROC/PRC preformance object to draw ROC/PRC curve
get_curve_input <- function(res_metrics, curve_type = 'ROC')
{
  
  if(curve_type == 'ROC'){
    
    model <- res_metrics$perf_ROC
    AUC <- res_metrics$perf_metrics['ROC_AUC'] %>% signif(.,4)
    
  }else{
    
    model <- res_metrics$perf_PRC
    AUC <- res_metrics$perf_metrics['PRC_AUC'] %>% signif(.,4)
    
  }
  
  if(class(model)[1] != 'performance'){
    v_x <- 0
    v_y <- 0
  }else{
    v_x <- model@x.values[[1]]
    v_y <- model@y.values[[1]]
  }
  
  res <- data.frame(x = v_x, y = v_y, AUC = AUC, row.names = NULL)
  res[is.na(res)] <- 0
  if(curve_type == 'ROC') colnames(res) <- c('FPR','TPR','AUC')
  if(curve_type == 'PRC') colnames(res) <- c('Recall','Precision','AUC')
  
  return(res)
}

## organize the result from 'get_evaluate_metrics' function
clean_res_metrcs <- function(res_metrics, sheetID, method){
  
  df_metrics <- res_metrics$perf_metrics
  df_metrics <- matrix(c(df_metrics,sheetID)) %>% t() %>% as.data.frame()
  rownames(df_metrics) <- NULL
  colnames(df_metrics) <- c(names(res_metrics$perf_metrics),'sheet')
  df_metrics$method <- method
  
  df_ROC <- get_curve_input(res_metrics = res_metrics, curve_type = 'ROC')
  df_ROC$method <- method
  df_ROC$sheet <- sheetID 
  
  df_PRC <- get_curve_input(res_metrics = res_metrics, curve_type = 'PRC')
  df_PRC$method <- method
  df_PRC$sheet <- sheetID
  
  res <- list(df_metrics=df_metrics,df_ROC=df_ROC,df_PRC=df_PRC)
}

######################
## load simulation  ##
######################

## load simulation data
prepare_input_data <- function(sheetID)
{
  
  exprMat <- openxlsx::read.xlsx("./data/ExpressionMatrix_100_slides.xlsx", 
                                 sheet = sheetID, colNames = F, startRow = 2)
  colnames(exprMat) <- paste0('Cell_',1:ncol(exprMat))
  rownames(exprMat) <- c(paste0('Lig',1:5),paste0('Rec',1:2),paste0('TF',1:3),paste0('TG',1:4))
  
  locaMat <- openxlsx::read.xlsx("./data/Lable_Coordinates_100_slides.xlsx", 
                                 sheet = sheetID, colNames = F, )
  locaMat <- locaMat[,-1]
  rownames(locaMat) <- colnames(exprMat)
  colnames(locaMat) <- c('dim_x','dim_y')
  
  annoMat <- openxlsx::read.xlsx("./data/ExpressionMatrix_100_slides.xlsx", 
                                 sheet = sheetID, colNames = F, rows = 1)
  annoMat <- rbind(colnames(exprMat),annoMat) %>% t() %>% as.data.frame()
  colnames(annoMat) <- c('Barcode','Cluster')
  rownames(annoMat) <- NULL
  annoMat$Cluster <- paste0("CT_",annoMat$Cluster)
  
  locaMat <- locaMat[!duplicated(locaMat),]
  annoMat <- annoMat[annoMat$Barcode %in% rownames(locaMat),]
  exprMat <- exprMat[colnames(exprMat) %in% rownames(locaMat)]
  
  res = list(exprMat=exprMat, locaMat=locaMat, annoMat=annoMat)
  return(res)
  
}
