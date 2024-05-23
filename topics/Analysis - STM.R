# Source: https://jovantrajceski.medium.com/structural-topic-modeling-with-r-part-i-2da2b353d362

library(topicmodels)
library(lda)
library(slam)
library(stm)
library(ggplot2)
library(dplyr)
library(tidytext)
library(furrr) # try to make it faster
plan(multicore)
library(tm) # Framework for text mining
library(tidyverse) # Data preparation and pipes %>%
library(ggplot2) # For plotting word frequencies
library(wordcloud) # Wordclouds!
library(Rtsne)
library(rsvd)
library(geometry)
library(NLP)
library(ldatuning) 

news <- read.csv("./Documents/Research/climate_nlp/global-stocktake-documents/ipccreportbodyANDipccmentionsbody.csv",stringsAsFactors=T)
news$year <- format(as.Date(news$date, format="%Y-%m-%d"),"%Y")
nrow(news)

# Check for NAs
sapply(news, function(x) sum(is.na(x)))

# Overview of original dataset
str(news)
sapply(news, typeof)

# randomly sample 5000 rows & remove unnecessary columns
set.seed(830)
news_sample <-news[sample(nrow(news), 5000), -c(1,3)]

# * default parameters
processed <- textProcessor(news_sample$text, metadata = news_sample,
                           lowercase = TRUE, #*
                           removestopwords = TRUE, #*
                           removenumbers = TRUE, #*
                           removepunctuation = TRUE, #*
                           stem = TRUE, #*
                           wordLengths = c(3,Inf), #*
                           sparselevel = 1, #*
                           language = "en", #*
                           verbose = TRUE, #*
                           onlycharacter = TRUE, # not def
                           striphtml = FALSE, #*
                           customstopwords = NULL, #*
                           v1 = FALSE) #*

# filter out terms that don’t appear in more than 10 documents, or appear in more than 500 documents
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh=10, upper.thresh=500)
# filters out 9000 out of 11000 words

docs <- out$documents
vocab <- out$vocab
meta <-out$meta

# Check levels
levels(meta$first_author)
levels(meta$doc_type_major)

# k = 15
set.seed(836)
system.time({
  STM_15 <- stm(documents = out$documents, vocab = out$vocab,
                K = 15, prevalence = ~ doc_type_major,
                max.em.its = 75, data = out$meta,
                init.type = "Spectral", verbose = FALSE
  )
})
plot(STM_15)

# k = 30
set.seed(836)
system.time({
  STM_30 <- stm(documents = out$documents, vocab = out$vocab,
                K = 30, prevalence = ~ doc_type_major,
                max.em.its = 75, data = out$meta,
                init.type = "Spectral", verbose = FALSE
  )
})
plot(STM_30)
STM_30$theta
apply(STM_30$theta, MARGIN=1, FUN=which.max)
STM_30$theta

# Adds topic labels to news samples
meta$topic <- apply(STM_30$theta, MARGIN=1, FUN=which.max)
write.csv(meta,"topics_STM.csv")

# Extracts the topics and labels
topic_labels<-labelTopics(STM_30)
topic_names<-topic_labels$prob
topic_names$combined <- apply(topic_names, 1, paste, collapse=", ")
write.csv(topic_names$combined,"topics_STM_labels.csv")

# Can I do this for the whole lot? Add labels to dataset -> then see if the breakdowns are skewed too?
# What does the research paper do? - does not use estimateeffect
# Where does EstimateEffect come in? - too confusing?
# Note this is just done on a sample - can do for all?

