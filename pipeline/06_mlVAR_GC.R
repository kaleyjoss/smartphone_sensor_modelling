library(mnet)
library(dplyr)
library(mlVAR)

args <- commandArgs(trailingOnly = TRUE)
name <- args[1]
brighten_dir = "/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/data/interim"
output_dir =  "/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/charts/mlVAR/trainval"


path <- file.path(brighten_dir, paste0(name, "_trainval_pca.csv"))
df <- read.csv(path)
pc_cols <- grep("^pc_", colnames(df), value = TRUE)
cat(pc_cols)
df$bin_clin2 <- as.integer(df$bin_clin + 1)
df$bin_clin2[df$bin_clin2 == 3] <- 2
table(df$bin_clin2)


clin <- mlVAR_GC(data = df,  
                        vars = pc_cols, 
                        idvar = "num_id", 
                        groups = "bin_clin2", 
                        test = "permutation", 
                        paired = FALSE, 
                        contemporaneous = "orthogonal", 
                        temporal = "orthogonal", 
                        nP = 1000, nCores = 12, pbar = TRUE)

l_outlist <- list()
l_outlist$GC <- v12day_clin


clin_g1 <- mlVAR(data = df[df$bin_clin2==1,],
                       vars = pc_cols,
                       estimator = "lmer",
                       idvar = "num_id",
                       contemporaneous = "orthogonal",
                       temporal = "orthogonal")

l_outlist$g1 <- clin_g1


clin_g2 <- mlVAR(data = df[df$bin_clin2==2,],
                       vars = pc_cols,
                       estimator = "lmer",
                       idvar = "num_id",
                       contemporaneous = "orthogonal",
                       temporal = "orthogonal")

l_outlist$g2 <- clin_g2

saveRDS(l_outlist, file=file.path(brighten_dir, paste0("mnet_output_binclin",name,".RDS")))


# Plot no-clinical-level at baseline
# P-values for the five parameter types:
pvalues = clin$Pval

# The observed group differences (i.e., the test statistics)
# Can be found in:
empdiff = clin$EmpDiffs

# Specifically, the difference is: group 1 - group 2
# For example:
groupdiff = clin$EmpDiffs$Phi_mean[1,2,]
# The true group difference was -0.4

library(qgraph)
# --- Check Results ---

output <- l_outlist$GC
out_g1 <- l_outlist$g1
out_g2 <- l_outlist$g2

# output$Runtime_min
# output$Pval$Lagged_fixed
#
# mean(output$Pval$Lagged_fixed<=0.05)
# sum(output$Pval$Lagged_fixed<=0.05)
#
# output$EmpDiffs

# For Tutorial
l_nets <- list()
l_nets$phi_group0 <- out_g1$results$Beta$mean[, , 1]
l_nets$phi_group1 <- out_g2$results$Beta$mean[, , 1]
l_nets$phi_diff <- output$EmpDiffs$Lagged_fixed[, , 1]
l_nets$phi_diffs_sig <- l_nets$phi_diff
l_nets$phi_diffs_sig[output$Pval$Lagged_fixed>0.05] <- 0

output$Pval$Lagged_fixed

titles <- c("Below Clinical Cutoff", "Above Clinical Cutoff",
                  "Differences: Group Low - Group High",
                  "Significant Differences")

# --- Make Figure ---

pdf(file.path(output_dir, paste0("Figure_binclin2_mnet",name,".pdf")), width = 18, height = 14)

layout_mat <- matrix(1:4, 2, 2)
layout(layout_mat)

short_labels <- gsub("pc_", "", pc_cols)


for(i in 1:4) qgraph(t(l_nets[[i]]),
                     layout = "circle",
                     edge.labels = TRUE,
                     title = titles[i],
                     theme = "colorblind",
                     maximum = 0.2,
                     mar = rep(8, 4),      # large mar gives room for long labels
                     labels = short_labels,  # short but readable
                     label.scale = FALSE,
                     label.cex = 1,
                     vsize = 15,           # smaller nodes
                     esize = 10, 
                     asize = 8, 
                     label.prop = 1.5,    # pushes label outside the node boundary
                     edge.label.cex = 0.7,
                     edge.label.bg = TRUE,
                     label.norm = "OOOOOOOOOOOOOOOO")  # reserves space for long labels

dev.off()
