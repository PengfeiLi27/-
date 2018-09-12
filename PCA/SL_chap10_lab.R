setwd("C:/Users/pxl4593/Desktop/machine learning/PCA")

states <- row.names(USArrests)
feature <- colnames(USArrests)

# calculate mean, var of each col
apply(USArrests,2,mean)
apply(USArrests,2,var)

# before PCA, data need to be standardized to have zero mean
pr.out <- prcomp(USArrests, scale = TRUE)
data_pca <- as.data.frame(pr.out$x)
pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)

plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", 
     ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
