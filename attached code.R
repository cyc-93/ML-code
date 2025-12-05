## read data and library packages ####
library(lme4)
library(readxl)
library(MuMIn)
library(sjPlot)
library(car)
library(visreg)
library(ggplot2)

### Fig.3c 
rm(list = ls())
data <- read.csv("correlation.csv")
data$StudyNo <- factor(data$StudyNo)
m1<-lmer(Biomass~Chlab+(1|StudyNo),data=data, weights = weight)
summary(m1)
c1 <- confint(m1,method="boot", nsim = 1000)
c1
fixef(m1)
ranef(m1)
coef(m1)
r.squaredGLMM(m1)
tab_model(m1)
Anova(m1)
point.size <- data$weight+2
p1 <- visreg(m1, "Chlab", gg=TRUE,ylab=expression("yi_biomass"),xlab=expression("Chlab"),line=list(col="black",linewidth = 0.8),fill=list(fill="gray",alpha=0.6),
             points=list(size=point.size, shape = 19, col="#2356a7",alpha=0.6)) +
  theme_classic() +
  theme(axis.ticks = element_line(linewidth = 0.5))+
  theme(axis.ticks.length = unit(0.15, "cm"))+
  theme(panel.border = element_rect(linetype = "solid", linewidth = 0.7, fill = NA))+
  scale_y_continuous(limits = c(-0.5, 1.8), breaks = seq(from = -1, to = 2, by = 0.5)) +
  scale_x_continuous(limits = c(-3, 3), breaks = seq(from = -3, to = 3, by = 1)) +
  geom_hline(yintercept =0,lty="dashed",color="gray",linewidth=0.8)+ #2.2  #纵横坐标轴加虚线
  theme(axis.text = element_text(size=18,color="black"),
        axis.line = element_blank(),
        axis.title = element_text(size=18,color="black"))+
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 5),vjust = 2))+
  theme(axis.text.x = element_text(margin = margin(t = 0, r = 0, b = 5, l = 0)))

p1

### Fig.3f
library(readxl)
library(piecewiseSEM)
library(lme4)
dat <- read_excel("SEM2.xlsx") 
data <- na.omit(dat)
data$StudyNo <- factor(data$StudyNo)


model1 <- lm(Biomass~MDA+H2O2,data)
summary(model1)
coefs(model1, standardize = 'scale')
beta_MDA <-  summary(model1)$coefficients[2, 1]
beta_H2O2 <- summary(model1)$coefficients[3, 1]
Oxidative <-   - (beta_MDA * data$MDA + beta_H2O2 * data$H2O2 )
data$Oxidative <- Oxidative

model2 <- lm(Biomass ~ SOD+CAT+POD,data)
summary(model2)
coefs(model2, standardize = 'scale')
beta_SOD <-  summary(model2)$coefficients[2, 1]
beta_CAT <- summary(model2)$coefficients[3, 1]
beta_POD <- summary(model2)$coefficients[4, 1]
Antioxidant <- beta_SOD * data$SOD + beta_CAT * data$CAT+ beta_POD * data$POD
data$Antioxidant <- Antioxidant

T.list <- list(
  lmer(Biomass ~ Oxidative+Antioxidant+Chlab+(1|StudyNo),weights=wi,
       data = data),
  lmer(Antioxidant~ Oxidative+(1|StudyNo),weights=wi,
       data = data),
  lmer(Chlab~  Oxidative+(1|StudyNo),weights=wi,
       data = data)
)
T.psem <- as.psem(T.list)
summary(T.psem)
basisSet(T.psem)
#conduct d-sep test
keeley.dsep3 <-dSep(T.psem)
keeley.dsep3
dSep(T.psem,conditioning=T)

#compute Fisher's C
fisherC(T.psem)

coefs(T.psem)
rsquared(T.psem)
AIC(T.psem)
fisherC(T.psem)
coefs(T.psem)
rsquared(T.psem)

p1 <- plot(T.psem)
p1




## Publication bias and Time lag-FigS2-FigS3 ####
library(dplyr)
library(tidyr)
library(metafor)
library(orchaRd)
library(openxlsx)
library(ggplot2)
library(patchwork)
###1Biomass P1 P2
rm(list=ls())
dat1 <- read.xlsx("Whole.xlsx", sheet = "1Biomass")
head(dat1)
##
##
## data for model
df_lnRR_pb1 <- dat1 %>% 
  filter(!is.na(SMD)) %>% 
  filter(!is.na(vSMD)) %>% 
  mutate(s_weights = 1/vSMD,
         sqrt_inv_eff_ss = sqrt((1/BControlN) + (1/BtreamentN)),
         inv_eff_ss = (1/BControlN) + (1/BtreamentN))
df_lnRR_pb1
##

## small-study effects
model_pub_bias11 <- rma.mv(yi = SMD, 
                           V = vSMD, 
                           mods = ~ 1 +
                             sqrt_inv_eff_ss, 
                           random = list(~1|StudyNo,
                                         ~1|EffectID),
                           data=df_lnRR_pb1, 
                           method="REML")
summary(model_pub_bias11)
r2_ml(model_pub_bias11)

funnel(model_pub_bias11, main="Funnel Plot")
p1=funnel(model_pub_bias11, level=c(90, 95, 99), shade=c("white","gray", "darkgray"), refline=0, atransf=exp,at=log(c(.10, .25, .5, 1, 2, 4, 10)))
+legend(1.2, 0.02, c("0.1 > p > 0.05", "0.05 > p >0.01", "< 0.01"), fill=c("white", "gray", "darkgray"))
p1


## time lag effects
model_pub_bias12 <- rma.mv(yi = SMD, 
                           V = vSMD, 
                           mods = ~ 1 +
                             year, 
                           random = list(~1|StudyNo,
                                         ~1|EffectID),
                           data=df_lnRR_pb1, 
                           method="REML")
summary(model_pub_bias12)
r2_ml(model_pub_bias12)
p2= ggplot(df_lnRR_pb1, aes(x = year, y = SMD, size = vSMD)) + geom_hline(yintercept = 0, linetype = "dashed") + 
  geom_abline(intercept = model_pub_bias12$beta[[1]], slope = model_pub_bias12$beta[[2]]) +
  geom_point(shape = 21, fill = "grey90") + 
  labs(x = "Publication year", y = "Effect size (Zr)", size ="Precision (1/SE)") + 
  guides(fill = "none", colour = "none") +
  theme(legend.position = c(0, 0.15), legend.justification = c(0, 1)) +
  theme(legend.direction = "horizontal") +
  # theme(legend.background = element_rect(fill = "white", colour = "black")) +
  theme(legend.background = element_blank()) +
  theme(axis.text.y = element_text(size = 10, colour = "black", hjust = 0.5, angle = 90))
p2


### Fig.S8
library(readxl)
library(ggplot2)
library(ggExtra) 
library(ggsci) 
library(dplyr)

rm(list = ls())
data <- read.csv("correlation.csv")
data <- data %>%
  filter(biomass >= -3 & biomass <= 4,
         size >= 0 & size <= 200)
data$group <- factor(data$group)
m1 <- lm(biomass ~ size, data = data)
summary(m1)
p <- ggplot(data, aes(x = size, y = biomass, color = group)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(
    data = data, aes(x = size, y = biomass),
    method = "lm", se = TRUE,
    color = "black", fill  = "grey80",
    linewidth = 1, alpha = 0.6
  ) +
  scale_color_manual(values = c("#e68b81", "#84c3b7", "#7da6c6")) +
  scale_x_continuous(limits = c(0, 200), breaks = seq(0, 200, 50)) +
  scale_y_continuous(limits = c(-3, 4), breaks = seq(-3, 4, 1)) +
  labs(x = "size", y = "biomass") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size=14, face="bold"),
    axis.text = element_text(size=12)
  )
p
p1 <- ggMarginal(p, type = "density", alpha = 0.7,
                 color = "grey20", groupFill = TRUE)
p1

ggsave("p1.pdf", p1, width = 5, height = 5, units = "in")
